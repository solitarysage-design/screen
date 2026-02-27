"""Standalone patch → candidates_fixed_v2_1.csv

Applies on top of output/20260226/candidates_fixed_v2.csv:
  1) is_financial fix    : 証券・商品先物取引業 の文字コード不一致を substring match で補正
  2) needs_manual_dividend_check 拡張 : high_yield_risk / split_mismatch / yf高利回り / fallback
  3) shares_missing fallback : yield_total NaN & yield_dps 存在 → DPS 代替、yield_used="dps_fallback"
  4) core_buyable_now / satellite_buyable_now 追加
  5) Assertions & validation log
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
INPUT_CSV    = OUTPUT_DIR / "candidates_fixed_v2.csv"
OUT_DIR      = OUTPUT_DIR

# ── Financial sector definitions ──────────────────────────────────────────────
# Exact strings — both 全角読点（、0x3001）and 半角中点（･ 0xff65）variants
FINANCIAL_EXACT = {
    "銀行業",
    "保険業",
    "その他金融業",
    "証券、商品先物取引業",   # 全角読点 (config)
    "証券･商品先物取引業",   # 半角中点 (J-Quants CSV実際の値)
}
# Substring fallback  ── 上記どちらにも当てはまらない場合
FINANCIAL_SUBSTRINGS = ("証券", "商品先物", "銀行", "保険")

CORE_YIELD_MIN  = 0.03
YF_HIGH_YIELD   = 0.045   # yfinance ソース & この利回り以上 → 手動確認
HIGH_YIELD_RISK = 0.06    # high_yield_risk フラグの閾値
CORE_TREND_MIN  = 4       # core_buyable_now の trend_score 閾値


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    """True for financial sector stocks (exact OR substring match)."""
    if _nan(sector):
        return False
    s = str(sector).strip()
    if s in FINANCIAL_EXACT:
        return True
    return any(sub in s for sub in FINANCIAL_SUBSTRINGS)


def _parse_flags(v) -> list[str]:
    """Parse semicolon-separated flag string or list."""
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if _nan(v) or str(v).strip() in ("", "nan", "[]"):
        return []
    return [x.strip() for x in str(v).split(";") if x.strip()]


# ── Core pass re-evaluator (minimal — only rechecks E condition + financial) ──
def _recompute_core_pass(row: pd.Series) -> tuple[bool, bool, bool, str]:
    """Re-run core_screen logic with fixed is_financial and updated yield_total."""
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

    # E: yield_total ONLY
    yield_total = _val(row.get("dividend_yield_fwd_total"))
    if yield_total is None or yield_total < CORE_YIELD_MIN:
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
        if yield_total is None or yield_total < CORE_YIELD_MIN:
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


def _make_drop_reason(row: pd.Series) -> str:
    """Rebuild drop_reason from core + satellite drop reasons + data_missing."""
    reasons: list[str] = []
    for field in ("core_drop_reasons", "satellite_drop_reasons"):
        dr = row.get(field)
        if _nan(dr):
            continue
        for r in str(dr).split(";"):
            r = r.strip()
            if r and r.lower() != "nan" and r not in reasons:
                reasons.append(r)

    # data_missing
    for col, key in [
        ("non_cut_years_verified",   "non_cut_years"),
        ("dividend_yield_fwd_total", "yield_total"),
        ("cfo_pos_5y_ratio",         "cfo_pos_ratio"),
        ("fcf_pos_5y_ratio",         "fcf_pos_ratio"),
        ("fcf_payout_3y",            "fcf_payout"),
    ]:
        if _nan(row.get(col)):
            tag = f"data_missing:{key}"
            if tag not in reasons:
                reasons.append(tag)

    return "; ".join(reasons) if reasons else "OK"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h_df = pd.read_csv(HOLDINGS_CSV, dtype=str)
    holdings_codes: set[str] = set(h_df["code_jquants_5digit"].str.strip().dropna())
    assert len(holdings_codes) == 21

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    df["code"] = df["code"].astype(str).str.strip()
    logger.info("Loaded v2 CSV: %d rows, %d cols", len(df), len(df.columns))

    # ── 1) is_financial fix ────────────────────────────────────────────────────
    old_fin = int(df["is_financial"].sum()) if "is_financial" in df.columns else 0
    df["is_financial"] = df["Sector33CodeName"].apply(_is_financial)
    new_fin = int(df["is_financial"].sum())
    logger.info(
        "is_financial: %d → %d (+%d, 証券セクター追加)",
        old_fin, new_fin, new_fin - old_fin,
    )
    # Show which sectors now newly flagged
    new_fin_sectors = (
        df[df["is_financial"] & (df["Sector33CodeName"].str.contains("証券", na=False) |
                                  df["Sector33CodeName"].str.contains("商品先物", na=False))]
        ["Sector33CodeName"].value_counts()
    )
    logger.info("  新たに追加された金融セクター:\n%s", new_fin_sectors.to_string())

    # ── 3) shares_missing → yield_total DPS fallback ──────────────────────────
    # Apply before core recompute so yield_total is fresh
    missing_yt = df["dividend_yield_fwd_total"].isna()
    has_ydps   = df["dividend_yield_fwd"].notna()
    fallback_mask = missing_yt & has_ydps

    n_fallback_holdings = int((fallback_mask & df["in_holdings"]).sum())
    logger.info(
        "shares_missing fallback 対象: %d件（うち保有株: %d件）",
        int(fallback_mask.sum()), n_fallback_holdings,
    )

    # Initialize yield_used
    df["yield_used"] = "none"
    # Pre-existing proper yield_total
    df.loc[~missing_yt, "yield_used"] = "total"

    # Apply fallback
    df.loc[fallback_mask, "dividend_yield_fwd_total"] = df.loc[fallback_mask, "dividend_yield_fwd"]
    df.loc[fallback_mask, "yield_used"] = "dps_fallback"

    # Add quality flag to fallback rows
    for idx in df.index[fallback_mask]:
        flags = _parse_flags(df.at[idx, "data_quality_flags"])
        if "yield_total_fallback_dps" not in flags:
            flags.append("yield_total_fallback_dps")
        df.at[idx, "data_quality_flags"] = "; ".join(flags)

    logger.info(
        "yield_used distribution: %s",
        df["yield_used"].value_counts().to_dict(),
    )
    logger.info(
        "dividend_yield_fwd_total after fallback: %d non-NaN (was %d)",
        int(df["dividend_yield_fwd_total"].notna().sum()),
        int((~missing_yt).sum()),
    )

    # ── 2) needs_manual_dividend_check 拡張 ───────────────────────────────────
    def _build_manual_reason(row: pd.Series) -> str | None:
        reasons: list[str] = []
        y_total = _val(row.get("dividend_yield_fwd_total"))
        y_dps   = _val(row.get("dividend_yield_fwd"))
        yield_v = y_total if y_total is not None else y_dps

        if row.get("high_yield_risk") is True:
            reasons.append("high_yield_risk")
        if row.get("yield_split_mismatch") is True:
            reasons.append("yield_split_mismatch")
        flags = _parse_flags(row.get("data_quality_flags"))
        if "yield_total_fallback_dps" in flags:
            reasons.append("yield_total_fallback_dps")
        src = str(row.get("div_source_used") or "")
        if src == "yf" and yield_v is not None and yield_v >= YF_HIGH_YIELD:
            reasons.append("yf_source_high_yield")

        return "; ".join(reasons) if reasons else None

    df["manual_check_reason"] = df.apply(_build_manual_reason, axis=1)

    # Expand needs_manual_dividend_check
    old_nmc = int(df["needs_manual_dividend_check"].sum())
    new_trigger = df["manual_check_reason"].notna()
    df["needs_manual_dividend_check"] = df["needs_manual_dividend_check"].astype(bool) | new_trigger
    new_nmc = int(df["needs_manual_dividend_check"].sum())
    logger.info(
        "needs_manual_dividend_check: %d → %d (+%d)",
        old_nmc, new_nmc, new_nmc - old_nmc,
    )
    logger.info(
        "  manual_check_reason内訳:\n%s",
        df["manual_check_reason"].dropna().str.split("; ").explode().value_counts().to_string(),
    )

    # ── Recompute core_pass / core_fin_pass ───────────────────────────────────
    logger.info("Recomputing core_pass for all %d rows...", len(df))
    core_results = df.apply(_recompute_core_pass, axis=1)
    df["core_pass"]         = [r[0] for r in core_results]
    df["core_fin_pass"]     = [r[1] for r in core_results]
    df["core_candidate"]    = [r[2] for r in core_results]
    df["core_drop_reasons"] = [r[3] for r in core_results]
    df["core_momo_pass"]    = (
        df["core_pass"]
        & df.get("tt_all_pass", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    )

    # ── 4) core_buyable_now / satellite_buyable_now ───────────────────────────
    df["core_buyable_now"] = (
        df["core_pass"]
        & (df["trend_score"].fillna(0).astype(int) >= CORE_TREND_MIN)
    )
    df["satellite_buyable_now"] = (
        df["satellite_pass"].fillna(False).astype(bool)
        & df.get("tt_all_pass", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    )

    # ── Rebuild drop_reason ───────────────────────────────────────────────────
    df["drop_reason"] = df.apply(_make_drop_reason, axis=1)

    # ── 6) Validation log ──────────────────────────────────────────────────────
    logger.info("=== Validation ===")

    # is_financial count
    n_fin = int(df["is_financial"].sum())
    logger.info("is_financial=True: %d (v2 was %d)", n_fin, old_fin)

    # core_pass に is_financial=True が混入していないこと
    fin_in_core = int((df["is_financial"] & df["core_pass"]).sum())
    assert fin_in_core == 0, f"FAIL: is_financial=True in core_pass: {fin_in_core}件"
    logger.info("core_pass に is_financial 混入: 0件 ✓")

    # needs_manual_dividend_check
    n_manual = int(df["needs_manual_dividend_check"].sum())
    n_high_yield = int((df["high_yield_risk"] == True).sum())
    assert n_manual >= n_high_yield, f"FAIL: manual({n_manual}) < high_yield({n_high_yield})"
    logger.info(
        "needs_manual_dividend_check: %d件（high_yield_risk %d件以上 ✓）",
        n_manual, n_high_yield,
    )

    # 保有株 shares_missing & fallback
    h_mask = df["in_holdings"] == True
    h_shares_miss = int(df.loc[h_mask, "data_quality_flags"].fillna("").str.contains("shares_missing").sum())
    h_fallback    = int((df.loc[h_mask, "yield_used"] == "dps_fallback").sum())
    logger.info("保有株21銘柄 shares_missing: %d件  dps_fallback適用: %d件", h_shares_miss, h_fallback)

    # drop_reason NaN
    n_dr_nan = int(df["drop_reason"].isna().sum())
    assert n_dr_nan == 0, f"FAIL: drop_reason NaN={n_dr_nan}"
    logger.info("drop_reason NaN: 0 ✓")

    # in_holdings count
    n_hold = int(df["in_holdings"].sum())
    assert n_hold == 21
    logger.info("in_holdings True: %d ✓", n_hold)

    # Summary
    n_core     = int(df["core_pass"].sum())
    n_buy_now  = int(df["core_buyable_now"].sum())
    n_fin_pass = int(df["core_fin_pass"].sum())
    n_sat      = int(df["satellite_pass"].sum()) if "satellite_pass" in df.columns else 0
    n_sat_buy  = int(df["satellite_buyable_now"].sum())
    logger.info(
        "core_pass=%d  core_buyable_now=%d  core_fin_pass=%d  "
        "satellite_pass=%d  satellite_buyable_now=%d",
        n_core, n_buy_now, n_fin_pass, n_sat, n_sat_buy,
    )

    # yield_used distribution
    logger.info("yield_used: %s", df["yield_used"].value_counts().to_dict())

    # ── 5) Output files ───────────────────────────────────────────────────────

    # candidates_fixed_v2_1.csv
    out_main = OUT_DIR / "candidates_fixed_v2_1.csv"
    df.to_csv(out_main, index=False, encoding="utf-8-sig")
    logger.info("candidates_fixed_v2_1.csv → %s (%d rows)", out_main, len(df))

    # manual_dividend_check_queue.csv
    queue_cols = [
        "code", "Name", "yield_used",
        "dividend_yield_fwd_total", "dividend_yield_fwd",
        "non_cut_years_verified", "fcf_payout_3y",
        "drop_reason", "data_quality_flags", "manual_check_reason",
        "div_source_used", "high_yield_risk", "yield_split_mismatch",
        "is_financial", "core_pass", "in_holdings",
    ]
    queue_df = df[df["needs_manual_dividend_check"] == True].copy()
    queue_cols_present = [c for c in queue_cols if c in queue_df.columns]
    queue_out = OUT_DIR / "manual_dividend_check_queue.csv"
    queue_df[queue_cols_present].to_csv(queue_out, index=False, encoding="utf-8-sig")
    logger.info("manual_dividend_check_queue.csv → %s (%d rows)", queue_out, len(queue_df))

    # holdings_debug_v2_1.csv
    debug_cols = [
        "code", "Name", "in_holdings", "is_financial",
        "core_pass", "core_fin_pass", "core_candidate", "core_buyable_now",
        "satellite_pass", "satellite_buyable_now",
        "drop_reason", "core_drop_reasons",
        "yield_used", "dividend_yield_fwd_total", "dividend_yield_fwd",
        "yield_split_mismatch", "non_cut_years_verified", "coverage_years",
        "cfo_pos_5y_ratio", "fcf_pos_5y_ratio", "fcf_payout_3y",
        "value_pass", "eps_fwd", "data_quality_flags",
        "needs_manual_dividend_check", "manual_check_reason",
        "rs_percentile", "trend_score", "tt_all_pass", "composite_score",
    ]
    debug_present = [c for c in debug_cols if c in df.columns]
    h_debug = df[df["in_holdings"] == True][debug_present].copy()
    h_merge = h_df[["code_jquants_5digit", "name_pdf", "account", "shares", "avg_cost_yen"]].merge(
        h_debug, left_on="code_jquants_5digit", right_on="code", how="left",
    )
    debug_out = OUT_DIR / "holdings_debug_v2_1.csv"
    h_merge.to_csv(debug_out, index=False, encoding="utf-8-sig")
    logger.info("holdings_debug_v2_1.csv → %s (%d rows)", debug_out, len(h_merge))

    # core_pass_top30_v2_1.csv
    score_col = "composite_score" if "composite_score" in df.columns else "rs_percentile"
    core_top = df[df["core_pass"]].sort_values(score_col, ascending=False).head(30)
    core_top.to_csv(OUT_DIR / "core_pass_top30_v2_1.csv", index=False, encoding="utf-8-sig")
    logger.info("core_pass_top30_v2_1.csv: %d rows", len(core_top))

    # core_buyable_now_top30.csv
    buy_top = df[df["core_buyable_now"]].sort_values(score_col, ascending=False).head(30)
    buy_top.to_csv(OUT_DIR / "core_buyable_now_top30.csv", index=False, encoding="utf-8-sig")
    logger.info("core_buyable_now_top30.csv: %d rows", len(buy_top))

    # ── Round-trip assertions ─────────────────────────────────────────────────
    df2 = pd.read_csv(out_main, encoding="utf-8-sig", low_memory=False)
    assert int(df2["in_holdings"].sum()) == 21
    assert int(df2["drop_reason"].isna().sum()) == 0
    fin_in_core2 = int((df2["is_financial"] & (df2["core_pass"] == True)).sum())
    assert fin_in_core2 == 0, f"Round-trip FAIL: is_financial in core_pass={fin_in_core2}"
    logger.info("=== All assertions passed ===")


if __name__ == "__main__":
    main()
