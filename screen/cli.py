"""CLI entry point — 増配バリュー × ミネルヴィニ スクリーナー."""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd

from screen.config import DEFAULT_LOOKBACK_DAYS, OUTPUT_DIR, UNKNOWN_POLICY
from screen.data.fundamentals import get_fundamentals
from screen.data.prices import get_prices, get_topix
from screen.data.universe import get_universe
from screen.features.breakouts import compute_breakouts
from screen.features.fundamentals_metrics import compute_fundamentals_metrics
from screen.features.relative_strength import compute_rs
from screen.features.technicals import compute_technicals
from screen.screens.minervini_gate import apply_minervini
from screen.screens.core_screen import apply_core_screen
from screen.screens.satellite_screen import apply_satellite_screen
from screen.screens.value_screen import compute_value_metrics
from screen.report import write_html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_date(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise click.BadParameter(f"Cannot parse date: {s}")


def _write_moomoo(codes: list[str], output_dir: Path) -> None:
    out = output_dir / "moomoo_watchlist.txt"
    out.write_text("\n".join(f"JP.{c}" for c in sorted(codes)), encoding="utf-8")
    logger.info("Moomoo watchlist: %s (%d tickers)", out, len(codes))


def _composite_score(row: pd.Series) -> float:
    rs = float(row.get("rs_percentile") or 0)
    eps = float(row.get("eps_score") or 0)
    bo = 5 if row.get("breakout_20d") else (3 if row.get("first_pullback") else 0)
    # Reward core fundamentals
    non_cut = min(float(row.get("non_cut_years") or 0), 10) * 0.5
    # 総額ベース利回りを優先、なければ DPS ベース
    yield_val = row.get("dividend_yield_fwd_total") or row.get("dividend_yield_fwd") or 0
    yield_bonus = 3 if yield_val >= 0.04 else 0
    return rs * 0.5 + eps * 3 + bo + non_cut + yield_bonus


def _load_holdings(holdings_csv: str) -> tuple[pd.DataFrame, set[str]]:
    """保有株 CSV を読み込み、J-Quants 5桁コードのセットを返す。

    Returns:
        (holdings_df, holdings_codes)
        holdings_codes: J-Quants 5桁コードのセット（ユニバースとのマッチに使用）
    """
    path = Path(holdings_csv)
    if not path.exists():
        raise FileNotFoundError(f"holdings_csv が見つかりません: {holdings_csv}")

    for enc in ("utf-8-sig", "utf-8", "cp932", "shift-jis"):
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise ValueError(f"holdings_csv のエンコーディングを特定できません: {holdings_csv}")
    logger.info("保有株CSV読み込み: %s (%d行)", path, len(df))

    codes: set[str] = set()

    if "code_jquants_5digit" in df.columns:
        for v in df["code_jquants_5digit"].dropna():
            v = str(v).strip()
            if v:
                codes.add(v)
        logger.info("保有株: code_jquants_5digit から %d件取得", len(codes))
    elif "code_4digit" in df.columns:
        for v in df["code_4digit"].dropna():
            v = str(v).strip()
            if v:
                codes.add(v + "0")  # J-Quants 5桁コードに変換（末尾0）
        logger.info("保有株: code_4digit+'0' から %d件取得", len(codes))
    else:
        raise ValueError("holdings_csv に code_jquants_5digit または code_4digit 列が必要です")

    # code_match 列を追加（後で merged との突合に使用）
    if "code_jquants_5digit" in df.columns:
        df["_code_match"] = df["code_jquants_5digit"].str.strip().fillna("")
    else:
        df["_code_match"] = df["code_4digit"].str.strip().fillna("") + "0"

    return df, codes


def _stringify_list_cols(df: pd.DataFrame) -> pd.DataFrame:
    """list/dict型セルを ; 区切り文字列に変換する（CSV出力用）。"""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], (list, dict)):
                df[col] = df[col].apply(
                    lambda v: "; ".join(str(x) for x in v) if isinstance(v, list)
                    else str(v) if v is not None else ""
                )
    return df


# Core screening に必須の列 → NaN なら data_missing:<key> を drop_reason に追記
_CORE_REQUIRED_COLS: list[tuple[str, str]] = [
    ("non_cut_years_verified",    "non_cut_years"),
    ("dividend_yield_fwd_total",  "yield_total"),
    ("cfo_pos_5y_ratio",          "cfo_pos_ratio"),
    ("fcf_pos_5y_ratio",          "fcf_pos_ratio"),
    ("fcf_payout_3y",             "fcf_payout"),
]


def _make_unified_drop_reason(row: pd.Series) -> str:
    """core_drop_reasons + satellite_drop_reasons + data_missing を統合した drop_reason。
    pass 銘柄は "OK"、落ちた銘柄は理由を "; " 区切りで返す（NaN/空文字は返さない）。
    """
    core_dr = row.get("core_drop_reasons")
    sat_dr = row.get("satellite_drop_reasons")
    reasons: list[str] = []
    for dr in [core_dr, sat_dr]:
        # None/NaN はスキップ（"nan" という文字列にしない）
        if dr is None or (isinstance(dr, float) and np.isnan(dr)):
            continue
        for r in str(dr).split(";"):
            r = r.strip()
            if r and r.lower() != "nan" and r not in reasons:
                reasons.append(r)

    # data_missing: core に必要な列が NaN の場合に追記
    for col, key in _CORE_REQUIRED_COLS:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            tag = f"data_missing:{key}"
            if tag not in reasons:
                reasons.append(tag)

    # "OK" for passing stocks — empty string would become NaN on CSV round-trip
    return "; ".join(reasons) if reasons else "OK"


@click.command()
@click.option("--asof", required=True, help="スクリーニング日 (YYYY-MM-DD)")
@click.option("--universe", default="prime_standard", type=click.Choice(["prime", "standard", "prime_standard", "growth", "all"]))
@click.option("--min_yield", default=0.03, type=float, help="最低配当利回り (Core モード)")
@click.option("--lookback_days", default=DEFAULT_LOOKBACK_DAYS, type=int)
@click.option("--rs_mode", default="universe_percentile", type=click.Choice(["universe_percentile", "topix"]))
@click.option("--export/--no-export", default=True)
@click.option("--explain", is_flag=True, default=False)
@click.option("--custom_csv", default=None)
@click.option("--holdings_csv", default=None, help="保有株マスターCSV（code_jquants_5digit または code_4digit 列が必要）")
@click.option("--unknown_policy", default=UNKNOWN_POLICY, type=click.Choice(["include", "exclude"]))
def run(asof, universe, min_yield, lookback_days, rs_mode, export, explain, custom_csv, holdings_csv, unknown_policy):
    """増配バリュー × ミネルヴィニ スクリーナー."""
    asof_date = _parse_date(asof)
    start_date = asof_date - timedelta(days=lookback_days)

    logger.info("=== Screener start | asof=%s | universe=%s ===", asof_date, universe)

    # ── Step 1: 保有株の読み込み ─────────────────────────────────────────
    holdings_df_raw: pd.DataFrame | None = None
    holdings_codes: set[str] = set()

    if holdings_csv:
        logger.info("Step 1a: 保有株CSV読み込み...")
        holdings_df_raw, holdings_codes = _load_holdings(holdings_csv)
        logger.info("保有株: %d銘柄", len(holdings_codes))

    # ── Step 2: Universe ─────────────────────────────────────────────────
    logger.info("Step 2: Fetching universe...")
    univ_df = get_universe(segment=universe, custom_csv=custom_csv)

    # 保有株を必ず universe に union する
    if holdings_codes:
        existing_codes = set(univ_df["Code"].tolist())
        new_h_codes = holdings_codes - existing_codes
        if new_h_codes:
            logger.info("保有株 union: ユニバース外の %d銘柄を追加", len(new_h_codes))
            extra_rows = pd.DataFrame({"Code": list(new_h_codes)})
            univ_df = pd.concat([univ_df, extra_rows], ignore_index=True)
        else:
            logger.info("保有株は全てユニバース内にあります")

    codes = univ_df["Code"].tolist()
    logger.info("Universe (保有株 union 後): %d codes", len(codes))

    # ── Step 3: Prices ───────────────────────────────────────────────────
    logger.info("Step 3: Fetching prices (%s → %s)...", start_date, asof_date)
    prices_df, excluded = get_prices(codes, start_date, asof_date)
    excluded_set = set(excluded)

    # 除外された保有株をログに記録
    excluded_holdings = holdings_codes & excluded_set
    if excluded_holdings:
        logger.warning("保有株のうち価格取得不可で除外: %s", sorted(excluded_holdings))

    active_codes = [c for c in codes if c not in excluded_set]
    logger.info("Prices OK: %d codes (除外: %d)", len(active_codes), len(excluded_set))

    # ── Step 4: Technicals ───────────────────────────────────────────────
    logger.info("Step 4: Computing technicals...")
    tech_df = compute_technicals(prices_df, asof_date)

    # ── Step 5: RS scores ────────────────────────────────────────────────
    logger.info("Step 5: Computing RS scores (mode=%s)...", rs_mode)
    topix_df = None
    if rs_mode == "topix":
        topix_df = get_topix(start_date, asof_date)
        if topix_df is None:
            logger.warning("TOPIX unavailable, using universe_percentile")
            rs_mode = "universe_percentile"
    rs_df = compute_rs(prices_df, topix=topix_df, mode=rs_mode, asof=asof_date)

    # ── Step 6: Minervini gate (pre-filter for Satellite) ────────────────
    logger.info("Step 6: Minervini gate...")
    tt_df = apply_minervini(tech_df, rs_df)

    # ── Step 7: Fundamentals ─────────────────────────────────────────────
    # Core screen は TT に依存しないため、全 active_codes + 保有株で取得する。
    # Satellite は satellite_screen.py 内で tt_all_pass を条件とする（別ルート）。
    # 初回は遅い（~26分）が diskcache 24h TTL で2回目以降は高速。
    tt_pass_codes = tt_df[tt_df["tt_all_pass"]]["code"].tolist()
    core_fund_codes = list(set(active_codes) | holdings_codes)
    logger.info(
        "Step 7: Fetching fundamentals (%d codes = %d active + %d holdings-only)...",
        len(core_fund_codes), len(active_codes),
        len(holdings_codes - set(active_codes)),
    )
    fund_df = get_fundamentals(core_fund_codes)

    # ── Step 8: Fundamental metrics ──────────────────────────────────────
    logger.info("Step 8: Computing fundamental metrics...")
    fm_df = compute_fundamentals_metrics(fund_df)

    # ── Step 9: Merge everything ─────────────────────────────────────────
    logger.info("Step 9: Merging...")
    sector_cols = [c for c in ["Sector33CodeName", "Sector17CodeName"] if c in univ_df.columns]
    univ_slim = univ_df[
        ["Code"] + sector_cols + (["Name"] if "Name" in univ_df.columns else [])
    ].rename(columns={"Code": "code"})

    merged = (
        tech_df
        .merge(rs_df[["code", "rs_percentile", "ret_3m", "ret_6m", "ret_9m", "ret_12m"]], on="code", how="left")
        .merge(tt_df, on="code", how="left")
        .merge(fm_df.drop(columns=["eps_q_list"], errors="ignore"), on="code", how="left")
        .merge(univ_slim, on="code", how="left")
    )

    # ── Step 10: Value metrics ────────────────────────────────────────────
    logger.info("Step 10: Computing value metrics...")
    merged = compute_value_metrics(merged)

    # ── Step 11: Core screen ─────────────────────────────────────────────
    logger.info("Step 11: Core screen...")
    merged = apply_core_screen(merged, unknown_policy=unknown_policy)

    # ── Step 12: Satellite screen ─────────────────────────────────────────
    logger.info("Step 12: Satellite screen...")
    merged = merged.merge(fm_df[["code", "eps_q_list"]], on="code", how="left")
    merged = apply_satellite_screen(merged, unknown_policy=unknown_policy)

    # ── Step 13: Breakouts ────────────────────────────────────────────────
    logger.info("Step 13: Computing breakouts...")
    bo_df = compute_breakouts(prices_df, asof_date)
    merged = merged.merge(bo_df, on="code", how="left")

    # ── Step 14: 保有株スタブ行の追加（価格取得不可で pipeline から落ちた銘柄） ──
    if holdings_codes:
        pipeline_codes = set(merged["code"].tolist())
        missing_holdings = holdings_codes - pipeline_codes
        if missing_holdings:
            logger.warning(
                "保有株のうちパイプラインに存在しない %d件をスタブ行として追加: %s",
                len(missing_holdings), sorted(missing_holdings),
            )
            stub_rows = pd.DataFrame({
                "code": list(missing_holdings),
                "core_pass": False,
                "satellite_pass": False,
                "core_drop_reasons": "no_price_data",
                "satellite_drop_reasons": "no_price_data",
            })
            merged = pd.concat([merged, stub_rows], ignore_index=True)

    # ── Step 15: in_holdings フラグ ───────────────────────────────────────
    merged["in_holdings"] = merged["code"].isin(holdings_codes) if holdings_codes else False

    # ── Step 16: Composite score ──────────────────────────────────────────
    merged["composite_score"] = merged.apply(_composite_score, axis=1)
    merged = merged.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # ── Step 17: 統合 drop_reason ─────────────────────────────────────────
    merged["drop_reason"] = merged.apply(_make_unified_drop_reason, axis=1)

    # ── Validation summary ───────────────────────────────────────────────
    n_core = int(merged["core_pass"].sum())
    n_sat = int(merged["satellite_pass"].sum())
    n_candidate = int(merged["core_candidate"].sum()) if "core_candidate" in merged.columns else 0
    n_momo = int(merged["core_momo_pass"].sum()) if "core_momo_pass" in merged.columns else 0
    logger.info(
        "Results: total=%d core=%d (momo=%d) satellite=%d core_candidate=%d",
        len(merged), n_core, n_momo, n_sat, n_candidate,
    )

    # non_cut_years_verified describe（max<=4 はバグ疑い）
    if "non_cut_years_verified" in merged.columns:
        ncy = merged["non_cut_years_verified"].dropna()
        if not ncy.empty:
            logger.info("non_cut_years_verified describe:\n%s", ncy.describe().to_string())
            if ncy.max() <= 4:
                logger.warning(
                    "non_cut_years_verified max=%.0f — 分割調整または配当履歴の実装を確認してください",
                    ncy.max(),
                )

    # dividend_yield_fwd_total describe
    if "dividend_yield_fwd_total" in merged.columns:
        ylds = merged["dividend_yield_fwd_total"].dropna()
        if not ylds.empty:
            logger.info("dividend_yield_fwd_total describe:\n%s", ylds.describe().to_string())
            high_yield = (ylds > 0.10).sum()
            if high_yield > 0:
                logger.warning("dividend_yield_fwd_total > 10%% の銘柄: %d件 — 分割未調整疑い", high_yield)

    # yield split mismatch
    mismatch_n = int(merged["yield_split_mismatch"].sum()) if "yield_split_mismatch" in merged.columns else 0
    logger.info("yield_split_mismatch 件数: %d", mismatch_n)

    # fcf_payout_3y describe
    payout_vals = merged["fcf_payout_3y"].dropna()
    if not payout_vals.empty:
        logger.info("fcf_payout_3y describe:\n%s", payout_vals.describe().to_string())
        med = payout_vals.median()
        if not (0.05 <= med <= 2.0):
            logger.warning("fcf_payout_3y median=%.4f が想定レンジ外 — 単位ズレを確認してください", med)

    # core_pass drop_reason top10（0件でなくても常に表示）
    logger.info("core_pass: %d件 / %d", n_core, len(merged))
    if "core_drop_reasons" in merged.columns:
        top_reasons = (
            merged["core_drop_reasons"]
            .dropna()
            .str.split("; ")
            .explode()
            .str.strip()
            .value_counts()
            .head(10)
        )
        logger.info("core_drop_reasons top10:\n%s", top_reasons.to_string())
    if n_core == 0:
        logger.warning("Core 0件通過 — 閾値またはデータを確認してください")
        if n_candidate > 0:
            logger.info(
                "Core candidates (coverage不足だが他条件OK): %d件 — "
                "needs_manual_dividend_check=True で出力に含まれます",
                n_candidate,
            )

    # ── Assertion F1: in_holdings 件数 ───────────────────────────────────
    n_in_holdings = int(merged["in_holdings"].sum()) if holdings_codes else 0
    logger.info("in_holdings True: %d行 (保有株 %d件)", n_in_holdings, len(holdings_codes))
    if holdings_codes and n_in_holdings != len(holdings_codes):
        logger.error(
            "in_holdings アサート失敗: %d件が True のはず, 実際=%d",
            len(holdings_codes), n_in_holdings,
        )
    else:
        logger.info("in_holdings アサートOK: %d件", n_in_holdings)

    # ── Assertion F2: 保有株の財務カバレッジ ─────────────────────────────
    if holdings_codes:
        h_mask = merged["in_holdings"]
        for col, key in _CORE_REQUIRED_COLS:
            if col in merged.columns:
                n_nan_h = int(merged.loc[h_mask, col].isna().sum())
                status = "OK" if n_nan_h == 0 else "WARNING"
                logger.info(
                    "[%s] 保有株 %s NaN: %d / %d",
                    status, col, n_nan_h, n_in_holdings,
                )

    # ── Assertion F3: drop_reason NaN チェック ───────────────────────────
    n_dr_nan = int(merged["drop_reason"].isna().sum())
    logger.info("drop_reason NaN: %d行 (0 が正常)", n_dr_nan)
    if n_dr_nan > 0:
        logger.error("drop_reason に NaN が %d行あります", n_dr_nan)

    # ── Assertion F4: data_missing 系 drop_reason 上位 ───────────────────
    dm_series = (
        merged["drop_reason"]
        .fillna("")
        .str.split("; ")
        .explode()
        .str.strip()
    )
    dm_counts = dm_series[dm_series.str.startswith("data_missing:")].value_counts().head(10)
    if not dm_counts.empty:
        logger.info("data_missing 系 drop_reason 上位:\n%s", dm_counts.to_string())
    else:
        logger.info("data_missing 系 drop_reason: 0件（全銘柄に財務データ取得済み）")

    # ── 保有株デバッグログ ────────────────────────────────────────────────
    if holdings_codes:
        in_h_codes = set(merged[merged["in_holdings"]]["code"].tolist())
        missing_h = holdings_codes - in_h_codes
        if missing_h:
            logger.error(
                "保有株アサート失敗: %d件が最終出力に存在しない → %s",
                len(missing_h), sorted(missing_h),
            )

        h_debug_cols = [
            "code", "Name",
            "dividend_yield_fwd_total", "non_cut_years_verified",
            "core_pass", "core_drop_reasons", "data_quality_flags",
        ]
        h_debug_cols = [c for c in h_debug_cols if c in merged.columns]
        h_rows = merged[merged["in_holdings"]][h_debug_cols]
        if not h_rows.empty:
            logger.info("保有株スクリーニング結果 (%d件):\n%s", len(h_rows), h_rows.to_string(index=False))

    # ── Explain mode ─────────────────────────────────────────────────────
    if explain:
        show_mask = (
            merged["core_pass"]
            | merged["satellite_pass"]
            | merged.get("core_candidate", pd.Series(False, index=merged.index))
        )
        for _, row in merged[show_mask].iterrows():
            print(f"\n{'='*60}")
            candidate_tag = " [CANDIDATE]" if row.get("core_candidate") else ""
            print(f"{row['code']} {row.get('Name','?')} | Score:{row['composite_score']:.1f}{candidate_tag}")
            yield_show = row.get("dividend_yield_fwd_total") or row.get("dividend_yield_fwd") or 0
            print(f"  RS:{row.get('rs_percentile','?'):.1f}%ile  EPS:{row.get('eps_score',0)}  Yield_total:{yield_show*100:.1f}%")
            print(
                f"  non_cut_verified:{row.get('non_cut_years_verified','?')}yr"
                f"(coverage:{row.get('coverage_years','?')}/{row.get('non_cut_years_required',5)})"
                f"  src:{row.get('div_source_used','?')}"
                f"  manual_check:{row.get('needs_manual_dividend_check','?')}"
            )
            print(f"  fcf_payout_3y:{row.get('fcf_payout_3y','?')}")
            print(f"  Core:{row.get('core_pass')} Candidate:{row.get('core_candidate')} drops={row.get('core_drop_reasons','')}")
            print(f"  Satellite:{row.get('satellite_pass')} drops={row.get('satellite_drop_reasons','')}")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(
        f"全候補: {len(merged)} | Core通過: {n_core} | "
        f"Core候補(coverage不足): {n_candidate} | Satellite通過: {n_sat}"
    )
    show_cols = [
        "code", "composite_score", "rs_percentile",
        "dividend_yield_fwd_total", "dividend_yield_fwd",
        "non_cut_years_verified", "non_cut_years_required",
        "coverage_years", "div_source_used",
        "needs_manual_dividend_check", "fcf_payout_3y",
        "core_pass", "core_candidate", "satellite_pass", "in_holdings",
    ]
    show_cols = [c for c in show_cols if c in merged.columns]
    core_candidate_col = merged.get("core_candidate", pd.Series(False, index=merged.index))
    core_or_sat = merged[merged["core_pass"] | merged["satellite_pass"] | core_candidate_col]
    if not core_or_sat.empty:
        print(core_or_sat[show_cols].head(20).to_string(index=False))

    # ── Export ────────────────────────────────────────────────────────────
    if export:
        # 実行日ごとにサブディレクトリを作成して上書きを防ぐ
        dated_dir = OUTPUT_DIR / asof_date.strftime("%Y%m%d")
        dated_dir.mkdir(parents=True, exist_ok=True)

        # Drop large list columns and stringify remaining list cols for CSV
        csv_df = merged.drop(columns=["eps_q_list"], errors="ignore")
        csv_df = _stringify_list_cols(csv_df)

        # ── A) universe_with_reasons.csv / candidates_fixed.csv（全銘柄）─
        required_cols = ["in_holdings", "core_pass", "satellite_pass", "drop_reason",
                         "core_drop_reasons", "satellite_drop_reasons"]
        present_required = [c for c in required_cols if c in csv_df.columns]
        universe_csv = dated_dir / "universe_with_reasons.csv"
        csv_df.to_csv(universe_csv, index=False, encoding="utf-8-sig")
        logger.info("universe_with_reasons.csv: %s (%d rows, 必須列: %s)",
                    universe_csv, len(csv_df), present_required)
        # candidates_fixed_v2.csv は universe_with_reasons.csv と同内容（仕様 E）
        fixed_csv = dated_dir / "candidates_fixed_v2.csv"
        csv_df.to_csv(fixed_csv, index=False, encoding="utf-8-sig")
        logger.info("candidates_fixed_v2.csv: %s", fixed_csv)

        # ── B) holdings_debug_fixed_v2.csv（保有株全行）─────────────────
        _holdings_debug_cols = [
            "code", "Name",
            "in_holdings", "core_pass", "core_candidate", "core_momo_pass",
            "satellite_pass",
            "drop_reason", "core_drop_reasons", "satellite_drop_reasons",
            "dividend_yield_fwd_total", "dividend_yield_fwd",
            "yield_split_mismatch", "fcf_payout_3y",
            "non_cut_years_verified", "non_cut_years_required",
            "coverage_years", "div_source_used", "needs_manual_dividend_check",
            "cfo_pos_5y_ratio", "fcf_pos_5y_ratio",
            "trend_score", "rs_percentile", "composite_score", "value_pass",
            "data_quality_flags",
        ]
        if holdings_df_raw is not None and not holdings_df_raw.empty:
            debug_cols_present = [c for c in _holdings_debug_cols if c in csv_df.columns]
            screen_slice = csv_df[debug_cols_present].copy()

            holdings_debug = holdings_df_raw.merge(
                screen_slice,
                left_on="_code_match",
                right_on="code",
                how="left",
            ).drop(columns=["_code_match"], errors="ignore")

            debug_csv = dated_dir / "holdings_debug_with_reasons.csv"
            holdings_debug.to_csv(debug_csv, index=False, encoding="utf-8-sig")
            logger.info("holdings_debug_with_reasons.csv: %s (%d rows)",
                        debug_csv, len(holdings_debug))
            # v2 ファイル（仕様 E）
            fixed_debug_csv = dated_dir / "holdings_debug_fixed_v2.csv"
            holdings_debug.to_csv(fixed_debug_csv, index=False, encoding="utf-8-sig")
            logger.info("holdings_debug_fixed_v2.csv: %s", fixed_debug_csv)

        # ── C) core_pass_top30.csv ─────────────────────────────────────
        core_df = csv_df[csv_df["core_pass"]].sort_values("composite_score", ascending=False).head(30)
        core_csv = dated_dir / "core_pass_top30.csv"
        core_df.to_csv(core_csv, index=False, encoding="utf-8-sig")
        logger.info("core_pass_top30.csv: %s (%d rows)", core_csv, len(core_df))

        # ── C2) core_candidates.csv（coverage不足だが他条件OK） ───────
        if "core_candidate" in csv_df.columns:
            cand_df = (
                csv_df[csv_df["core_candidate"]]
                .sort_values("composite_score", ascending=False)
                .head(50)
            )
            cand_csv = dated_dir / "core_candidates.csv"
            cand_df.to_csv(cand_csv, index=False, encoding="utf-8-sig")
            logger.info(
                "core_candidates.csv: %s (%d rows, needs_manual_dividend_check=True)",
                cand_csv, len(cand_df),
            )

        # ── D) satellite_pass_top20.csv ────────────────────────────────
        sat_df = csv_df[csv_df["satellite_pass"]].sort_values("composite_score", ascending=False).head(20)
        sat_csv = dated_dir / "satellite_pass_top20.csv"
        sat_df.to_csv(sat_csv, index=False, encoding="utf-8-sig")
        logger.info("satellite_pass_top20.csv: %s (%d rows)", sat_csv, len(sat_df))

        _write_moomoo(
            (core_df["code"].tolist() + sat_df["code"].tolist()),
            dated_dir,
        )

        # HTML report (core_pass or satellite_pass)
        report_df = merged[merged["core_pass"] | merged["satellite_pass"]].copy()
        html_path = write_html(report_df, asof_date, dated_dir)
        logger.info("HTML: %s", html_path)

    logger.info("=== Screener complete ===")


if __name__ == "__main__":
    run()
