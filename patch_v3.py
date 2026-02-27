"""Standalone patch → candidates_fixed_v3.csv

Applies on top of candidates_fixed_v2.csv:
  Stage A : yfinance batch dividend fetch for all stocks missing yield data
  Fin-sec : is_financial flag; core_pass=False for banks/insurance/other-fin
            core_fin_pass = new alternative pass for financial sector
  error_flags : stmt_fetch_failed:no_jquants_auth for holdings with missing J-Quants data
  drop_reason : fully re-computed (no NaN, no literal "nan")
  Assertions  : in_holdings==21, drop_reason NaN==0
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
HOLDINGS_CSV = Path("C:/Users/solit/Downloads/holdings_extracted_20260224.csv")
INPUT_CSV    = Path("C:/Users/solit/projects/screen/output/candidates_fixed_v2.csv")
OUT_DIR      = Path("C:/Users/solit/projects/screen/output")

# ── Financial sectors (J-Quants Sector33 labels) ─────────────────────────────
FINANCIAL_SECTOR33 = {"銀行業", "保険業", "その他金融業", "証券、商品先物取引業"}

# ── Core thresholds (mirrors core_screen.py) ─────────────────────────────────
CORE_NON_CUT_MIN   = 5
CORE_YIELD_MIN     = 0.03
CORE_CFO_POS_MIN   = 0.80
CORE_FCF_POS_MIN   = 0.60
CORE_FCF_PAYOUT_NC = 0.70  # non-cyclical
CORE_FCF_PAYOUT_CY = 0.60  # cyclical

CYCLICAL_SECTOR33 = {
    "鉄鋼", "非鉄金属", "石油・石炭製品", "化学", "海運業", "空運業", "鉱業", "建設業",
}

# Columns whose NaN → data_missing:key in drop_reason
_CORE_REQUIRED_COLS: list[tuple[str, str]] = [
    ("non_cut_years_verified",   "non_cut_years"),
    ("dividend_yield_fwd_total", "yield_total"),
    ("cfo_pos_5y_ratio",         "cfo_pos_ratio"),
    ("fcf_pos_5y_ratio",         "fcf_pos_ratio"),
    ("fcf_payout_3y",            "fcf_payout"),
]


# ── Stage A: yfinance batch yield fetch ──────────────────────────────────────
def _jq_to_yf(code: str) -> str:
    """J-Quants 5-digit → yfinance ticker  e.g. '14140' → '1414.T'"""
    return code[:4] + ".T"


def fetch_yield_yfinance(
    codes: list[str],
    prices: dict[str, float],
    batch_size: int = 200,
) -> dict[str, float]:
    """Batch-download 1-year dividend data from yfinance; return {code: annual_yield}."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — Stage A skipped")
        return {}

    code_to_yf = {c: _jq_to_yf(c) for c in codes}
    yf_to_code = {v: k for k, v in code_to_yf.items()}
    tickers_all = list(code_to_yf.values())

    results: dict[str, float] = {}
    total_batches = (len(tickers_all) + batch_size - 1) // batch_size

    for i in range(0, len(tickers_all), batch_size):
        batch = tickers_all[i: i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(
            "Stage A yfinance batch %d/%d (%d tickers)...",
            batch_num, total_batches, len(batch),
        )
        try:
            raw = yf.download(
                batch,
                period="1y",
                actions=True,
                auto_adjust=False,
                progress=False,
            )
            if raw is None or raw.empty:
                continue

            for ticker in batch:
                code = yf_to_code.get(ticker)
                if not code:
                    continue
                price = prices.get(code)
                if not price or price <= 0:
                    continue
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        # multi-ticker: columns = (field, ticker)
                        divs = raw.get(("Dividends", ticker), pd.Series(dtype=float))
                    else:
                        divs = raw.get("Dividends", pd.Series(dtype=float))

                    if divs is None:
                        continue
                    annual = float(divs.dropna().sum())
                    if annual > 0 and not np.isnan(annual):
                        results[code] = annual / price
                except Exception:
                    pass
        except Exception as e:
            logger.warning("yfinance batch %d failed: %s", batch_num, e)
        time.sleep(0.5)  # gentle rate limiting

    logger.info("Stage A: got yield for %d / %d codes", len(results), len(codes))
    return results


# ── Core screening helpers ────────────────────────────────────────────────────
def _nan(v) -> bool:
    """True if v is None or NaN."""
    if v is None:
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


def _recompute_core_pass(row: pd.Series) -> tuple[bool, bool, bool, str]:
    """Re-evaluate core_pass, core_fin_pass, core_candidate, core_drop_reasons.

    Returns (core_pass, core_fin_pass, core_candidate, core_drop_reasons_str).
    """
    sector = str(row.get("Sector33CodeName") or "")
    financial = sector in FINANCIAL_SECTOR33
    cyclical  = sector in CYCLICAL_SECTOR33
    payout_cap = CORE_FCF_PAYOUT_CY if cyclical else CORE_FCF_PAYOUT_NC

    # Coverage / non-cut
    non_cut = _val(row.get("non_cut_years_verified")) or _val(row.get("non_cut_years"))
    cov = _val(row.get("coverage_years")) or 0
    cov_int = int(cov)
    cov_insuf = cov_int < CORE_NON_CUT_MIN

    fund_drops: list[str] = []

    # A: non-cut years
    if non_cut is None or non_cut < CORE_NON_CUT_MIN:
        fund_drops.append("A_non_cut_years")
    if cov_insuf and "A_non_cut_years" not in fund_drops:
        fund_drops.append("A_coverage_insufficient")

    # B/C/D: CFO/FCF (non-financial only)
    if not financial:
        cfo = _val(row.get("cfo_pos_5y_ratio"))
        if cfo is None or cfo < CORE_CFO_POS_MIN:
            fund_drops.append("B_cfo_pos_ratio")
        fcf_pos = _val(row.get("fcf_pos_5y_ratio"))
        if fcf_pos is None or fcf_pos < CORE_FCF_POS_MIN:
            fund_drops.append("C_fcf_pos_ratio")
        payout = _val(row.get("fcf_payout_3y"))
        if payout is None or payout > payout_cap:
            fund_drops.append("D_fcf_payout")
        if row.get("fcf_payout_hard_fail") is True:
            fund_drops.append("D_fcf_hard_fail")
    else:
        fund_drops.append("fin_sector_excluded_from_core")

    # E: yield
    yield_val = _val(row.get("dividend_yield_fwd_total")) or _val(row.get("dividend_yield_fwd"))
    if yield_val is None or yield_val < CORE_YIELD_MIN:
        fund_drops.append("E_div_yield")

    # F: value_pass
    if not row.get("value_pass", False):
        fund_drops.append("F_value_pass")

    core_pass = (not financial) and len(fund_drops) == 0

    non_cov_drops = [
        d for d in fund_drops
        if d not in ("A_non_cut_years", "A_coverage_insufficient", "fin_sector_excluded_from_core")
    ]
    core_candidate = (not core_pass) and (not financial) and cov_insuf and len(non_cov_drops) == 0

    # core_fin_pass: financial sector alternative
    fin_drops: list[str] = []
    if financial:
        if non_cut is None or non_cut < CORE_NON_CUT_MIN:
            fin_drops.append("A_non_cut_years")
        if cov_insuf and "A_non_cut_years" not in fin_drops:
            fin_drops.append("A_coverage_insufficient")
        if yield_val is None or yield_val < CORE_YIELD_MIN:
            fin_drops.append("E_div_yield")
        if not row.get("value_pass", False):
            fin_drops.append("F_value_pass")
        eps_fwd = _val(row.get("eps_fwd"))
        if eps_fwd is None:
            fin_drops.append("G_eps_unknown")
        elif eps_fwd <= 0:
            fin_drops.append("G_eps_nonpositive")

    core_fin_pass = financial and len(fin_drops) == 0

    all_drops = fund_drops + ([f"fin:{r}" for r in fin_drops] if financial else [])
    return core_pass, core_fin_pass, core_candidate, "; ".join(all_drops) if all_drops else ""


def _make_drop_reason(row: pd.Series) -> str:
    """Full drop_reason: core_drop_reasons + satellite_drop_reasons + data_missing + error_flags.
    Never NaN, never empty string (uses 'OK' for fully-passing stocks).
    """
    reasons: list[str] = []

    for field in ["core_drop_reasons", "satellite_drop_reasons"]:
        dr = row.get(field)
        if dr is None or (isinstance(dr, float) and np.isnan(dr)):
            continue
        for r in str(dr).split(";"):
            r = r.strip()
            if r and r.lower() != "nan" and r not in reasons:
                reasons.append(r)

    # data_missing for NaN core columns
    for col, key in _CORE_REQUIRED_COLS:
        if _nan(row.get(col)):
            tag = f"data_missing:{key}"
            if tag not in reasons:
                reasons.append(tag)

    # error_flags
    ef = row.get("error_flags")
    if isinstance(ef, list):
        for f in ef:
            if f and f not in reasons:
                reasons.append(f)
    elif isinstance(ef, str) and ef and ef.lower() != "nan":
        for f in ef.split(";"):
            f = f.strip()
            if f and f not in reasons:
                reasons.append(f)

    return "; ".join(reasons) if reasons else "OK"


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    h_df = pd.read_csv(HOLDINGS_CSV, dtype=str)
    holdings_codes: set[str] = set(h_df["code_jquants_5digit"].str.strip().dropna())
    logger.info("Holdings: %d codes", len(holdings_codes))
    assert len(holdings_codes) == 21, f"Expected 21 holdings, got {len(holdings_codes)}"

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    df["code"] = df["code"].astype(str).str.strip()
    logger.info("Loaded v2 CSV: %d rows, %d cols", len(df), len(df.columns))

    # ── Fix in_holdings ──────────────────────────────────────────────────────
    df["in_holdings"] = df["code"].isin(holdings_codes)
    assert int(df["in_holdings"].sum()) == 21, "in_holdings assert failed"
    logger.info("in_holdings: %d True", int(df["in_holdings"].sum()))

    # ── Add is_financial ─────────────────────────────────────────────────────
    df["is_financial"] = df["Sector33CodeName"].isin(FINANCIAL_SECTOR33)
    n_fin = int(df["is_financial"].sum())
    logger.info("Financial stocks: %d", n_fin)
    logger.info("  sectors: %s",
        df[df["is_financial"]]["Sector33CodeName"].value_counts().to_dict())

    # ── Add error_flags column (inherit from v2 if present, else init) ───────
    if "error_flags" not in df.columns:
        df["error_flags"] = [[] for _ in range(len(df))]
    else:
        # Convert stringified lists back if needed
        def _parse_ef(v):
            if isinstance(v, list):
                return v
            if _nan(v) or str(v).strip() in ("", "nan", "[]"):
                return []
            return [x.strip() for x in str(v).split(";") if x.strip()]
        df["error_flags"] = df["error_flags"].apply(_parse_ef)

    # Mark holdings stocks with missing J-Quants data as stmt_fetch_failed
    h_missing_jq_mask = df["in_holdings"] & df["cfo_pos_5y_ratio"].isna()
    n_missing_jq = int(h_missing_jq_mask.sum())
    logger.info("Holdings with missing J-Quants CFO/FCF data: %d / 21", n_missing_jq)
    for idx in df.index[h_missing_jq_mask]:
        ef = list(df.at[idx, "error_flags"])
        if "stmt_fetch_failed:no_jquants_auth" not in ef:
            ef.append("stmt_fetch_failed:no_jquants_auth")
        df.at[idx, "error_flags"] = ef

    # ── Stage A: yfinance broad yield fetch ──────────────────────────────────
    # Fetch for ALL stocks missing dividend_yield_fwd (DPS-based approximate)
    missing_yield_mask = df["dividend_yield_fwd"].isna()
    missing_yield_codes = df.loc[missing_yield_mask, "code"].tolist()
    logger.info(
        "Stage A: %d stocks missing dividend_yield_fwd — fetching from yfinance...",
        len(missing_yield_codes),
    )

    prices_dict = dict(zip(df["code"], df["price"].apply(lambda v: _val(v))))

    yf_yield = fetch_yield_yfinance(missing_yield_codes, prices_dict)

    # Fill dividend_yield_fwd where NaN
    n_filled = 0
    for code, y in yf_yield.items():
        mask = df["code"] == code
        idx_list = df.index[mask].tolist()
        if idx_list and _nan(df.at[idx_list[0], "dividend_yield_fwd"]):
            df.at[idx_list[0], "dividend_yield_fwd"] = y
            # Mark yield source
            ef = list(df.at[idx_list[0], "error_flags"])
            # add quality flag (not error) — mark as yf_yield_approx
            n_filled += 1
    logger.info(
        "Stage A: filled dividend_yield_fwd for %d stocks (was %d, now %d non-NaN)",
        n_filled,
        int((~missing_yield_mask).sum()),
        int(df["dividend_yield_fwd"].notna().sum()),
    )

    # For stocks still without any yield after Stage A: mark yield_missing_source
    # (not an error_flag, but data_quality_flag)
    still_no_yield = df["dividend_yield_fwd"].isna() & df["dividend_yield_fwd_total"].isna()
    logger.info(
        "Stocks with NO yield after Stage A: %d (will get data_missing:yield in drop_reason)",
        int(still_no_yield.sum()),
    )

    # ── value_pass re-evaluation for stocks that got new yield data ──────────
    # value_pass uses PER/PBR/FCF which don't depend on yield, so no change needed.
    # But high_yield_risk threshold should use updated yield.
    df["high_yield_risk"] = df.apply(
        lambda r: bool(
            (_val(r.get("dividend_yield_fwd_total")) or _val(r.get("dividend_yield_fwd")) or 0)
            > 0.06
        ),
        axis=1,
    )

    # ── Re-run core_pass / core_fin_pass for all rows ────────────────────────
    logger.info("Re-computing core_pass / core_fin_pass for all %d rows...", len(df))
    core_results = df.apply(_recompute_core_pass, axis=1)
    df["core_pass"]         = [r[0] for r in core_results]
    df["core_fin_pass"]     = [r[1] for r in core_results]
    df["core_candidate"]    = [r[2] for r in core_results]
    df["core_drop_reasons"] = [r[3] for r in core_results]
    df["core_momo_pass"]    = df["core_pass"] & df["tt_all_pass"].fillna(False).astype(bool)

    n_core     = int(df["core_pass"].sum())
    n_fin_pass = int(df["core_fin_pass"].sum())
    n_sat      = int(df["satellite_pass"].sum()) if "satellite_pass" in df.columns else 0
    logger.info(
        "core_pass=%d / core_fin_pass=%d / satellite_pass=%d",
        n_core, n_fin_pass, n_sat,
    )

    # Check: no financial stock should be in core_pass
    fin_in_core = df[df["is_financial"] & df["core_pass"]]
    if not fin_in_core.empty:
        logger.error("BUG: %d financial stocks still in core_pass!", len(fin_in_core))
    else:
        logger.info("OK: 0 financial stocks in core_pass")

    # ── Re-compute drop_reason ────────────────────────────────────────────────
    df["drop_reason"] = df.apply(_make_drop_reason, axis=1)
    n_dr_nan = int(df["drop_reason"].isna().sum())
    assert n_dr_nan == 0, f"drop_reason still has {n_dr_nan} NaN rows"
    logger.info("drop_reason NaN: 0 ✓")

    # ── Validation F ─────────────────────────────────────────────────────────
    logger.info("=== Validation ===")
    logger.info("in_holdings True: %d (expected 21)", int(df["in_holdings"].sum()))

    h_mask = df["in_holdings"]
    check_cols = [
        ("dividend_yield_fwd_total", "yield_total"),
        ("dividend_yield_fwd",       "yield_dps"),
        ("non_cut_years_verified",   "non_cut_years"),
        ("cfo_pos_5y_ratio",         "cfo_pos_ratio"),
        ("fcf_pos_5y_ratio",         "fcf_pos_ratio"),
        ("fcf_payout_3y",            "fcf_payout"),
    ]
    for col, key in check_cols:
        if col in df.columns:
            n_nan = int(df.loc[h_mask, col].isna().sum())
            status = "OK" if n_nan == 0 else "WARNING"
            logger.info("[%s] 保有株 %s NaN: %d / 21", status, col, n_nan)

    logger.info(
        "dividend_yield_fwd non-NaN: %d / %d (was 217 in v2)",
        int(df["dividend_yield_fwd"].notna().sum()), len(df),
    )
    logger.info(
        "dividend_yield_fwd_total non-NaN: %d / %d (was 143 in v2)",
        int(df["dividend_yield_fwd_total"].notna().sum()), len(df),
    )

    # data_missing top
    dm = (
        df["drop_reason"]
        .str.split("; ")
        .explode()
        .str.strip()
    )
    dm_counts = dm[dm.str.startswith("data_missing:")].value_counts().head(8)
    if not dm_counts.empty:
        logger.info("data_missing top:\n%s", dm_counts.to_string())

    # stmt_fetch_failed top
    sf_counts = dm[dm.str.startswith("stmt_fetch_failed:")].value_counts().head(5)
    if not sf_counts.empty:
        logger.info("stmt_fetch_failed top:\n%s", sf_counts.to_string())

    # ── Stringify list columns for CSV ────────────────────────────────────────
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], list):
                df[col] = df[col].apply(
                    lambda v: "; ".join(str(x) for x in v) if isinstance(v, list) else (v or "")
                )

    # ── Save candidates_fixed_v3.csv ─────────────────────────────────────────
    # Ensure required output columns exist
    for col in ["core_fin_pass", "is_financial", "error_flags"]:
        if col not in df.columns:
            df[col] = ""

    out_v3 = OUT_DIR / "candidates_fixed_v3.csv"
    df.to_csv(out_v3, index=False, encoding="utf-8-sig")
    logger.info("candidates_fixed_v3.csv → %s (%d rows)", out_v3, len(df))

    # ── Save holdings_debug_fixed_v3.csv ─────────────────────────────────────
    debug_cols = [
        "code", "in_holdings", "is_financial",
        "core_pass", "core_fin_pass", "core_candidate", "core_momo_pass",
        "satellite_pass", "drop_reason", "core_drop_reasons",
        "dividend_yield_fwd_total", "dividend_yield_fwd",
        "yield_split_mismatch", "non_cut_years_verified", "coverage_years",
        "cfo_pos_5y_ratio", "fcf_pos_5y_ratio", "fcf_payout_3y",
        "value_pass", "eps_fwd", "data_quality_flags", "error_flags",
        "rs_percentile", "tt_all_pass", "composite_score",
    ]
    debug_cols_present = [c for c in debug_cols if c in df.columns]
    h_debug_screen = df[df["in_holdings"]][debug_cols_present].copy()

    h_merge = h_df[["code_jquants_5digit", "name_pdf", "account", "shares", "avg_cost_yen"]].merge(
        h_debug_screen,
        left_on="code_jquants_5digit",
        right_on="code",
        how="left",
    )
    out_h = OUT_DIR / "holdings_debug_fixed_v3.csv"
    h_merge.to_csv(out_h, index=False, encoding="utf-8-sig")
    logger.info("holdings_debug_fixed_v3.csv → %s (%d rows)", out_h, len(h_merge))

    # ── Save top lists ────────────────────────────────────────────────────────
    score_col = "composite_score" if "composite_score" in df.columns else "rs_percentile"

    core_top = (
        df[df["core_pass"]]
        .sort_values(score_col, ascending=False)
        .head(30)
    )
    core_top.to_csv(OUT_DIR / "core_pass_top30.csv", index=False, encoding="utf-8-sig")
    logger.info("core_pass_top30.csv: %d rows", len(core_top))

    fin_top = (
        df[df["core_fin_pass"]]
        .sort_values(score_col, ascending=False)
        .head(30)
    )
    fin_top.to_csv(OUT_DIR / "core_fin_pass_top30.csv", index=False, encoding="utf-8-sig")
    logger.info("core_fin_pass_top30.csv: %d rows", len(fin_top))

    if "satellite_pass" in df.columns:
        sat_top = (
            df[df["satellite_pass"]]
            .sort_values(score_col, ascending=False)
            .head(20)
        )
        sat_top.to_csv(OUT_DIR / "satellite_pass_top20.csv", index=False, encoding="utf-8-sig")
        logger.info("satellite_pass_top20.csv: %d rows", len(sat_top))

    # ── Final assertions ──────────────────────────────────────────────────────
    df2 = pd.read_csv(out_v3, encoding="utf-8-sig", low_memory=False)
    assert int(df2["in_holdings"].sum()) == 21, "FAIL: in_holdings"
    assert int(df2["drop_reason"].isna().sum()) == 0, "FAIL: drop_reason NaN"
    logger.info("=== All assertions passed ===")
    logger.info(
        "Summary: core_pass=%d / core_fin_pass=%d / satellite_pass=%d / total=%d",
        int(df2["core_pass"].sum()),
        int(df2["core_fin_pass"].sum()),
        int(df2.get("satellite_pass", pd.Series(dtype=bool)).sum()) if "satellite_pass" in df2.columns else 0,
        len(df2),
    )


if __name__ == "__main__":
    main()
