"""Standalone patch → candidates_fixed_v2_2.csv

Applies on top of output/20260226/candidates_fixed_v2_1.csv:
  A) drop_reason を core / satellite に分離
  B) TT系を info_flags に隔離（core判定から排除）
  C) 利回りの根拠明示 + total_div_fwd 合成 + effective_yield
  D) coverage_years キャップ (upper=10)
  E) Name/Sector 欠損補完（保有2銘柄ハードコード）
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
import os
PROJECT_DIR  = Path(os.environ.get("SCREEN_PROJECT_DIR", "C:/Users/solit/projects/screen"))
OUTPUT_DIR   = Path(os.environ.get("SCREEN_OUTPUT_DIR", str(PROJECT_DIR / "output" / "20260226")))
HOLDINGS_CSV = Path(os.environ.get("SCREEN_HOLDINGS_CSV", "C:/Users/solit/Downloads/holdings_extracted_20260224.csv"))
INPUT_CSV    = OUTPUT_DIR / "candidates_fixed_v2_1.csv"
OUT_DIR      = OUTPUT_DIR

# ── Constants ─────────────────────────────────────────────────────────────────
FINANCIAL_EXACT = {
    "銀行業",
    "保険業",
    "その他金融業",
    "証券、商品先物取引業",
    "証券･商品先物取引業",
}
FINANCIAL_SUBSTRINGS = ("証券", "商品先物", "銀行", "保険")

CORE_YIELD_MIN   = 0.03
HIGH_YIELD_RISK  = 0.06
CORE_TREND_MIN   = 4

# Name/Sector manual fill for holdings with NaN
MANUAL_FILL = {
    "51610": {"Name": "西川ゴム工業", "Sector33CodeName": "ゴム製品"},
    "69320": {"Name": "遠藤照明", "Sector33CodeName": "電気機器"},
}


# ── Helpers (from v2_1) ──────────────────────────────────────────────────────
def _nan(v) -> bool:
    if v is None:
        return True
    if v is pd.NA:
        return True
    try:
        return bool(np.isnan(float(v)))
    except (TypeError, ValueError):
        return False


def _val(v) -> float | None:
    if _nan(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_financial(sector) -> bool:
    if _nan(sector):
        return False
    s = str(sector).strip()
    if s in FINANCIAL_EXACT:
        return True
    return any(sub in s for sub in FINANCIAL_SUBSTRINGS)


def _parse_flags(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if _nan(v) or str(v).strip() in ("", "nan", "[]"):
        return []
    return [x.strip() for x in str(v).split(";") if x.strip()]


# ── Core pass re-evaluator (uses effective_yield instead of dividend_yield_fwd_total) ──
def _recompute_core_pass(row: pd.Series) -> tuple[bool, bool, bool, str]:
    """Re-run core_screen logic with effective_yield for condition E."""
    from screen.config import CYCLICAL_SECTOR33

    financial = bool(row.get("is_financial", False))
    sector = str(row.get("Sector33CodeName") or "")
    cyclical = sector in CYCLICAL_SECTOR33
    payout_cap = 0.60 if cyclical else 0.70

    non_cut = _val(row.get("non_cut_years_verified")) or _val(row.get("non_cut_years"))
    cov = int(_val(row.get("coverage_years")) or 0)
    cov_insuf = cov < 5

    fund_drops: list[str] = []

    # A
    if non_cut is None or non_cut < 5:
        fund_drops.append("A_non_cut_years")
    if cov_insuf and "A_non_cut_years" not in fund_drops:
        fund_drops.append("A_coverage_insufficient")

    # B/C/D
    if not financial:
        cfo = _val(row.get("cfo_pos_5y_ratio"))
        if cfo is None or cfo < 0.80:
            fund_drops.append("B_cfo_pos_ratio")
        fcf_pos = _val(row.get("fcf_pos_5y_ratio"))
        if fcf_pos is None or fcf_pos < 0.60:
            fund_drops.append("C_fcf_pos_ratio")
        payout = _val(row.get("fcf_payout_3y"))
        if payout is None or payout > payout_cap:
            fund_drops.append("D_fcf_payout")
        if row.get("fcf_payout_hard_fail") is True:
            fund_drops.append("D_fcf_hard_fail")
    else:
        fund_drops.append("fin_sector_excluded_from_core")

    # E: use effective_yield (new)
    ey = _val(row.get("effective_yield"))
    if ey is None or ey < CORE_YIELD_MIN:
        fund_drops.append("E_div_yield")

    # F
    if not row.get("value_pass", False):
        fund_drops.append("F_value_pass")

    core_pass = (not financial) and len(fund_drops) == 0

    non_cov_drops = [
        d for d in fund_drops
        if d not in ("A_non_cut_years", "A_coverage_insufficient",
                     "fin_sector_excluded_from_core")
    ]
    core_candidate = (
        not core_pass and not financial and cov_insuf and len(non_cov_drops) == 0
    )

    # core_fin_pass
    fin_drops: list[str] = []
    if financial:
        if non_cut is None or non_cut < 5:
            fin_drops.append("A_non_cut_years")
        if cov_insuf and "A_non_cut_years" not in fin_drops:
            fin_drops.append("A_coverage_insufficient")
        if ey is None or ey < CORE_YIELD_MIN:
            fin_drops.append("E_div_yield")
        if not row.get("value_pass", False):
            fin_drops.append("F_value_pass")
        eps = _val(row.get("eps_fwd"))
        if eps is None:
            fin_drops.append("G_eps_unknown")
        elif eps <= 0:
            fin_drops.append("G_eps_nonpositive")

    core_fin_pass = financial and len(fin_drops) == 0

    all_drops = fund_drops + ([f"fin:{r}" for r in fin_drops] if financial else [])
    return core_pass, core_fin_pass, core_candidate, "; ".join(all_drops) if all_drops else ""


# ── drop_reason builders ─────────────────────────────────────────────────────
def _build_drop_reason_core(row: pd.Series) -> str:
    """Build drop_reason_core from core conditions only."""
    if row.get("core_pass") is True:
        return "OK"

    reasons: list[str] = []

    # From core_drop_reasons (already recomputed)
    cdr = row.get("core_drop_reasons")
    if not _nan(cdr):
        for r in str(cdr).split(";"):
            r = r.strip()
            if r and r.lower() != "nan":
                reasons.append(r)

    # data_missing tags relevant to core
    for col, key in [
        ("non_cut_years_verified",   "non_cut_years"),
        ("effective_yield",          "yield"),
        ("cfo_pos_5y_ratio",         "cfo_pos_ratio"),
        ("fcf_pos_5y_ratio",         "fcf_pos_ratio"),
        ("fcf_payout_3y",            "fcf_payout"),
    ]:
        if _nan(row.get(col)):
            tag = f"data_missing:{key}"
            if tag not in reasons:
                reasons.append(tag)

    return "; ".join(reasons) if reasons else "OK"


def _build_drop_reason_satellite(row: pd.Series) -> str:
    """Build drop_reason_satellite from satellite conditions only."""
    if row.get("satellite_pass") is True:
        return "OK"

    reasons: list[str] = []
    sdr = row.get("satellite_drop_reasons")
    if not _nan(sdr):
        for r in str(sdr).split(";"):
            r = r.strip()
            if r and r.lower() != "nan":
                reasons.append(r)

    return "; ".join(reasons) if reasons else "OK"


def _build_info_flags(row: pd.Series) -> str:
    """Build info_flags — informational tags that don't affect pass/fail."""
    flags: list[str] = []
    tt_pass = row.get("tt_all_pass")
    if tt_pass is not True and tt_pass is not np.True_:
        flags.append("info_tt_not_all_pass")
    return "; ".join(flags) if flags else ""


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    h_df = pd.read_csv(HOLDINGS_CSV, dtype=str)
    holdings_codes: set[str] = set(h_df["code_jquants_5digit"].str.strip().dropna())
    assert len(holdings_codes) == 21, f"Expected 21 holdings, got {len(holdings_codes)}"

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    df["code"] = df["code"].astype(str).str.strip()
    logger.info("Loaded v2_1 CSV: %d rows, %d cols", len(df), len(df.columns))

    # ── Step 2: E) Name/Sector 欠損補完 ──────────────────────────────────────
    for code, fill in MANUAL_FILL.items():
        mask = df["code"] == code
        if mask.any():
            for col, val in fill.items():
                df.loc[mask, col] = val
            logger.info("Filled Name/Sector for code %s: %s", code, fill)

    # Assert: all holdings have Name and Sector
    h_mask = df["in_holdings"] == True
    n_hold = int(h_mask.sum())
    assert n_hold == 21, f"in_holdings count: {n_hold} != 21"
    name_nan_hold = int(df.loc[h_mask, "Name"].isna().sum())
    sector_nan_hold = int(df.loc[h_mask, "Sector33CodeName"].isna().sum())
    assert name_nan_hold == 0, f"Holdings with Name NaN: {name_nan_hold}"
    assert sector_nan_hold == 0, f"Holdings with Sector NaN: {sector_nan_hold}"
    logger.info("Name/Sector 欠損補完完了 — 保有株 Name NaN: 0, Sector NaN: 0 ✓")

    # ── Step 3: D) coverage_years キャップ ───────────────────────────────────
    old_max_cov = df["coverage_years"].max()
    df["coverage_years"] = df["coverage_years"].clip(upper=10)
    new_max_cov = df["coverage_years"].max()
    logger.info("coverage_years cap: max %s → %s", old_max_cov, new_max_cov)

    # ── Step 4: C) 利回りの根拠明示 + forward構築 ────────────────────────────

    # Recalculate is_financial (in case sector was just filled)
    df["is_financial"] = df["Sector33CodeName"].apply(_is_financial)

    # dividend_yield_total_actual
    price = df["price"]
    shares = df["net_shares_latest"]
    market_cap = price * shares  # per-share price * shares outstanding

    df["dividend_yield_total_actual"] = np.where(
        market_cap.notna() & (market_cap != 0) & df["total_div_actual"].notna(),
        df["total_div_actual"] / market_cap,
        np.nan,
    )

    # total_div_fwd synthesis
    # Original total_div_fwd is all NaN, but keep the logic general
    orig_fwd_present = df["total_div_fwd"].notna()
    can_synth = df["total_div_fwd"].isna() & df["dps_fwd"].notna() & df["net_shares_latest"].notna()

    df.loc[can_synth, "total_div_fwd"] = df.loc[can_synth, "dps_fwd"] * df.loc[can_synth, "net_shares_latest"]

    # yield_basis
    df["yield_basis"] = "none"
    df.loc[orig_fwd_present, "yield_basis"] = "fwd"
    df.loc[can_synth & ~orig_fwd_present, "yield_basis"] = "dps_fallback"
    # actual-only: has total_div_actual but no fwd
    actual_only = (
        df["total_div_actual"].notna()
        & (df["yield_basis"] == "none")
    )
    df.loc[actual_only, "yield_basis"] = "actual"

    # dividend_yield_total_fwd
    df["dividend_yield_total_fwd"] = np.where(
        market_cap.notna() & (market_cap != 0) & df["total_div_fwd"].notna(),
        df["total_div_fwd"] / market_cap,
        np.nan,
    )

    # effective_yield
    df["effective_yield"] = np.nan
    fwd_mask = df["yield_basis"].isin(["fwd", "dps_fallback"])
    df.loc[fwd_mask, "effective_yield"] = df.loc[fwd_mask, "dividend_yield_total_fwd"]
    actual_mask = df["yield_basis"] == "actual"
    df.loc[actual_mask, "effective_yield"] = df.loc[actual_mask, "dividend_yield_total_actual"]
    # yield_basis=="none" → effective_yield stays NaN

    # high_yield_risk (refresh with effective_yield)
    df["high_yield_risk"] = df["effective_yield"] > HIGH_YIELD_RISK

    logger.info(
        "yield_basis distribution: %s",
        df["yield_basis"].value_counts().to_dict(),
    )
    logger.info(
        "effective_yield: non-NaN=%d, median=%.4f",
        int(df["effective_yield"].notna().sum()),
        df["effective_yield"].median() if df["effective_yield"].notna().any() else 0,
    )

    # ── Step 5: core_pass / satellite_pass 再計算 ────────────────────────────
    logger.info("Recomputing core_pass for all %d rows...", len(df))
    core_results = df.apply(_recompute_core_pass, axis=1)
    df["core_pass"]         = [r[0] for r in core_results]
    df["core_fin_pass"]     = [r[1] for r in core_results]
    df["core_candidate"]    = [r[2] for r in core_results]
    df["core_drop_reasons"] = [r[3] for r in core_results]

    # core_momo_pass
    df["core_momo_pass"] = (
        df["core_pass"]
        & df["tt_all_pass"].fillna(False).astype(bool)
    )

    # core_buyable_now / satellite_buyable_now
    df["core_buyable_now"] = (
        df["core_pass"]
        & (df["trend_score"].fillna(0).astype(int) >= CORE_TREND_MIN)
    )
    df["satellite_buyable_now"] = (
        df["satellite_pass"].fillna(False).astype(bool)
        & df["tt_all_pass"].fillna(False).astype(bool)
    )

    # satellite_pass is NOT recomputed — kept as-is from v2_1

    # ── Step 6: A) drop_reason をモード別に分離 ──────────────────────────────
    df["drop_reason_core"] = df.apply(_build_drop_reason_core, axis=1)
    df["drop_reason_satellite"] = df.apply(_build_drop_reason_satellite, axis=1)

    # ── Step 7: B) info_flags ────────────────────────────────────────────────
    df["info_flags"] = df.apply(_build_info_flags, axis=1)

    # Keep legacy drop_reason for backwards compat (core + satellite merged)
    df["drop_reason"] = df.apply(
        lambda row: "OK" if row["core_pass"] or row["satellite_pass"] else
        "; ".join(filter(None, [
            row["drop_reason_core"] if row["drop_reason_core"] != "OK" else "",
            row["drop_reason_satellite"] if row["drop_reason_satellite"] != "OK" else "",
        ])) or "OK",
        axis=1,
    )

    # ── Step 8: 検証ログ（assert） ───────────────────────────────────────────
    logger.info("=== Validation ===")

    # 1. core_pass==True → drop_reason_core=="OK"
    core_pass_bad = df[df["core_pass"] & (df["drop_reason_core"] != "OK")]
    assert len(core_pass_bad) == 0, (
        f"FAIL: core_pass=True but drop_reason_core!='OK': {len(core_pass_bad)} rows\n"
        f"{core_pass_bad[['code','drop_reason_core']].to_string()}"
    )
    logger.info("core_pass=True → drop_reason_core='OK': 0 violations ✓")

    # 2. satellite_pass==True → drop_reason_satellite=="OK"
    sat_pass_bad = df[df["satellite_pass"].fillna(False).astype(bool) & (df["drop_reason_satellite"] != "OK")]
    assert len(sat_pass_bad) == 0, (
        f"FAIL: satellite_pass=True but drop_reason_satellite!='OK': {len(sat_pass_bad)} rows\n"
        f"{sat_pass_bad[['code','drop_reason_satellite']].to_string()}"
    )
    logger.info("satellite_pass=True → drop_reason_satellite='OK': 0 violations ✓")

    # 3. yield_basis 件数内訳
    yb_counts = df["yield_basis"].value_counts()
    logger.info("yield_basis breakdown:\n%s", yb_counts.to_string())

    # 4. in_holdings → Name/Sector non-NaN
    h_mask = df["in_holdings"] == True
    assert int(df.loc[h_mask, "Name"].isna().sum()) == 0, "Holdings with Name NaN"
    assert int(df.loc[h_mask, "Sector33CodeName"].isna().sum()) == 0, "Holdings with Sector NaN"
    logger.info("in_holdings → Name/Sector 非NaN ✓")

    # 5. in_holdings count
    assert int(h_mask.sum()) == 21, f"in_holdings={int(h_mask.sum())} != 21"
    logger.info("in_holdings=True: 21 ✓")

    # 6. drop_reason_core / drop_reason_satellite にNaN無し
    n_core_nan = int(df["drop_reason_core"].isna().sum())
    n_sat_nan = int(df["drop_reason_satellite"].isna().sum())
    assert n_core_nan == 0, f"drop_reason_core NaN: {n_core_nan}"
    assert n_sat_nan == 0, f"drop_reason_satellite NaN: {n_sat_nan}"
    logger.info("drop_reason_core NaN: 0, drop_reason_satellite NaN: 0 ✓")

    # Summary
    n_core     = int(df["core_pass"].sum())
    n_buy_now  = int(df["core_buyable_now"].sum())
    n_fin_pass = int(df["core_fin_pass"].sum())
    n_sat      = int(df["satellite_pass"].fillna(False).astype(bool).sum())
    n_sat_buy  = int(df["satellite_buyable_now"].sum())
    logger.info(
        "core_pass=%d  core_buyable_now=%d  core_fin_pass=%d  "
        "satellite_pass=%d  satellite_buyable_now=%d",
        n_core, n_buy_now, n_fin_pass, n_sat, n_sat_buy,
    )

    # coverage_years
    logger.info("coverage_years: max=%s", df["coverage_years"].max())

    # ── Step 9: 出力 ─────────────────────────────────────────────────────────

    # 1. candidates_fixed_v2_2.csv
    out_main = OUT_DIR / "candidates_fixed_v2_2.csv"
    df.to_csv(out_main, index=False, encoding="utf-8-sig")
    logger.info("candidates_fixed_v2_2.csv → %s (%d rows, %d cols)", out_main, len(df), len(df.columns))

    # 2. holdings_debug_v2_2.csv
    debug_cols = [
        "code", "Name", "in_holdings", "is_financial",
        "core_pass", "core_fin_pass", "core_candidate", "core_buyable_now",
        "satellite_pass", "satellite_buyable_now",
        "drop_reason_core", "drop_reason_satellite", "info_flags",
        "yield_basis", "dividend_yield_total_actual", "dividend_yield_total_fwd", "effective_yield",
        "dividend_yield_fwd_total", "dividend_yield_fwd",
        "total_div_fwd", "total_div_actual", "dps_fwd", "net_shares_latest",
        "high_yield_risk",
        "non_cut_years_verified", "coverage_years",
        "cfo_pos_5y_ratio", "fcf_pos_5y_ratio", "fcf_payout_3y",
        "value_pass", "eps_fwd", "data_quality_flags",
        "needs_manual_dividend_check", "manual_check_reason",
        "rs_percentile", "trend_score", "tt_all_pass", "composite_score",
        "drop_reason",
    ]
    debug_present = [c for c in debug_cols if c in df.columns]
    h_debug = df[df["in_holdings"] == True][debug_present].copy()
    h_merge = h_df[["code_jquants_5digit", "name_pdf", "account", "shares", "avg_cost_yen"]].merge(
        h_debug, left_on="code_jquants_5digit", right_on="code", how="left",
    )
    debug_out = OUT_DIR / "holdings_debug_v2_2.csv"
    h_merge.to_csv(debug_out, index=False, encoding="utf-8-sig")
    logger.info("holdings_debug_v2_2.csv → %s (%d rows)", debug_out, len(h_merge))

    # 3. manual_dividend_check_queue_v2_2.csv
    queue_cols = [
        "code", "Name", "yield_basis", "effective_yield",
        "dividend_yield_total_actual", "dividend_yield_total_fwd",
        "dividend_yield_fwd_total", "dividend_yield_fwd",
        "non_cut_years_verified", "fcf_payout_3y",
        "drop_reason_core", "drop_reason_satellite",
        "data_quality_flags", "manual_check_reason",
        "div_source_used", "high_yield_risk", "yield_split_mismatch",
        "is_financial", "core_pass", "satellite_pass", "in_holdings",
    ]
    queue_df = df[df["needs_manual_dividend_check"] == True].copy()
    queue_cols_present = [c for c in queue_cols if c in queue_df.columns]
    queue_out = OUT_DIR / "manual_dividend_check_queue_v2_2.csv"
    queue_df[queue_cols_present].to_csv(queue_out, index=False, encoding="utf-8-sig")
    logger.info("manual_dividend_check_queue_v2_2.csv → %s (%d rows)", queue_out, len(queue_df))

    # ── Round-trip assertions ─────────────────────────────────────────────────
    df2 = pd.read_csv(out_main, encoding="utf-8-sig", low_memory=False)
    assert int(df2["in_holdings"].sum()) == 21
    assert int(df2["drop_reason_core"].isna().sum()) == 0
    assert int(df2["drop_reason_satellite"].isna().sum()) == 0
    core_ok_check = df2[df2["core_pass"] == True]
    assert (core_ok_check["drop_reason_core"] == "OK").all(), "Round-trip: core_pass=True with non-OK drop_reason_core"
    logger.info("=== All assertions passed ===")


if __name__ == "__main__":
    main()
