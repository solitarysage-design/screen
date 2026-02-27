#!/usr/bin/env python3
"""patch_v2_3 — Dividend history 10-year expansion + manual check fix.

Reads candidates_fixed_v2_2.csv, re-fetches dividend history via yfinance
for stocks with coverage < 6, then applies:
  A) coverage_years expansion to up to 10 years via yfinance
  B) coverage_short → non_cut_years=NaN (not false-negative "reduced")
  C) non_cut_years_verified = bool(coverage >= required+1)
  D) Tighter needs_manual_dividend_check (yield anomaly only)
  E) Output to /mnt/data/
  F) Validation logs
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

# ── Paths ──────────────────────────────────────────────────────────────────
import os
PROJECT_DIR = Path(os.environ.get("SCREEN_PROJECT_DIR", "C:/Users/solit/projects/screen"))
OUTPUT_DIR = Path(os.environ.get("SCREEN_OUTPUT_DIR", str(PROJECT_DIR / "output" / "20260226")))
HOLDINGS_CSV = Path(os.environ.get("SCREEN_HOLDINGS_CSV", "C:/Users/solit/Downloads/holdings_extracted_20260224.csv"))
INPUT_CSV = OUTPUT_DIR / "candidates_fixed_v2_2.csv"
OUT_DIR = OUTPUT_DIR

# ── Constants ──────────────────────────────────────────────────────────────
NON_CUT_YEARS_REQUIRED = 5
CORE_YIELD_MIN = 0.03
HIGH_YIELD_RISK_THRESHOLD = 0.06
MIN_COVERAGE_TARGET = 6  # aim for at least 6 years

FINANCIAL_EXACT = {
    "銀行業", "保険業", "その他金融業",
    "証券、商品先物取引業", "証券･商品先物取引業",
}
FINANCIAL_SUBSTRINGS = ("証券", "商品先物", "銀行", "保険")

_CUT_TOL_ABS_JPY = 1.0
_CUT_TOL_REL = 0.02
_NON_CUT_MAX_Y = 10
_YF_DELAY_SEC = 0.25


# ── Helpers ────────────────────────────────────────────────────────────────
def _nan(v) -> bool:
    if v is None or v is pd.NA:
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


# ── A) yfinance dividend history fetcher ───────────────────────────────────
def _fetch_yf_dividends(code5: str, max_retries: int = 2) -> dict | None:
    """Fetch annual DPS from yfinance with robust split handling.

    Returns dict: records=[(year,dps),...], coverage, has_splits, split_adjusted
    or None on failure.
    """
    for attempt in range(max_retries):
        result = _fetch_yf_dividends_once(code5)
        if result is not None:
            return result
        if attempt < max_retries - 1:
            time.sleep(1.5)
    return None


def _fetch_yf_dividends_once(code5: str) -> dict | None:
    import yfinance as yf

    code4 = code5[:-1] if len(code5) == 5 and code5.endswith("0") else code5
    ticker_sym = f"{code4}.T"

    try:
        ticker = yf.Ticker(ticker_sym)
        hist = ticker.history(period="max", auto_adjust=False, actions=True)

        if hist.empty or "Dividends" not in hist.columns:
            return None

        raw_divs = hist["Dividends"].dropna()
        raw_divs = raw_divs[raw_divs > 0]
        if raw_divs.empty:
            return None

        # ── Split handling with bad-data filter ──
        splits = pd.Series(dtype=float)
        if "Stock Splits" in hist.columns:
            splits = hist["Stock Splits"].dropna()
            splits = splits[(splits > 0) & (splits >= 0.001) & (splits <= 10000)]
        has_splits = not splits.empty

        if has_splits:
            adj_dict: dict = {}
            for div_date, raw_dps in raw_divs.items():
                future_splits = splits[splits.index > div_date]
                factor = float(future_splits.prod()) if not future_splits.empty else 1.0
                if factor < 1e-4 or factor > 1e5:
                    factor = 1.0
                adj_dict[div_date] = raw_dps * factor
            adj_divs = pd.Series(adj_dict, dtype=float)
            split_adjusted = True
        else:
            adj_divs = raw_divs.copy()
            split_adjusted = False

        # ── Resample to calendar year ──
        try:
            annual = adj_divs.resample("YE").sum()
        except Exception:
            annual = adj_divs.resample("A").sum()
        annual = annual[annual > 0]
        if annual.empty:
            return None

        records = [(int(dt.year), float(dps)) for dt, dps in annual.items()]
        records = sorted(records, key=lambda x: x[0], reverse=True)
        records = records[:10]

        return {
            "records": records,
            "coverage": len(records),
            "has_splits": has_splits,
            "split_adjusted": split_adjusted,
        }
    except Exception as exc:
        logger.debug("yfinance %s: %s", code5, str(exc)[:80])
        return None


# ── Dividend-cut detection ─────────────────────────────────────────────────
def _is_div_cut(curr: float, prev: float) -> bool:
    if prev <= 0:
        return False
    diff = prev - curr
    if diff <= 0:
        return False
    return diff > _CUT_TOL_ABS_JPY and diff / prev > _CUT_TOL_REL


def _compute_non_cut_years(records: list[tuple[int, float]]) -> float | None:
    """From (year, dps) desc list, count consecutive non-cut years."""
    if len(records) < 2:
        return None
    count = 0
    for i in range(len(records) - 1):
        curr_dps, prev_dps = records[i][1], records[i + 1][1]
        if curr_dps <= 0 or prev_dps <= 0:
            break
        if _is_div_cut(curr_dps, prev_dps):
            break
        count += 1
    return float(min(count, _NON_CUT_MAX_Y))


# ── Core pass recompute ───────────────────────────────────────────────────
def _recompute_core_pass(row: pd.Series) -> tuple[bool, bool, bool, str]:
    from screen.config import CYCLICAL_SECTOR33

    financial = bool(row.get("is_financial", False))
    sector = str(row.get("Sector33CodeName") or "")
    cyclical = sector in CYCLICAL_SECTOR33
    payout_cap = 0.60 if cyclical else 0.70

    non_cut = _val(row.get("non_cut_years"))
    cov = int(_val(row.get("coverage_years")) or 0)
    div_short = "div_history_short" in str(row.get("data_quality_flags", ""))

    fund_drops: list[str] = []

    # A — coverage insufficient uses data_missing, not A_non_cut_years
    if div_short:
        fund_drops.append("data_missing:div_history")
    elif non_cut is None or non_cut < NON_CUT_YEARS_REQUIRED:
        fund_drops.append("A_non_cut_years")

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

    # E
    ey = _val(row.get("effective_yield"))
    if ey is None or ey < CORE_YIELD_MIN:
        fund_drops.append("E_div_yield")

    # F
    if not row.get("value_pass", False):
        fund_drops.append("F_value_pass")

    core_pass = (not financial) and len(fund_drops) == 0

    non_cov_drops = [
        d for d in fund_drops
        if d not in ("data_missing:div_history", "A_non_cut_years",
                     "A_coverage_insufficient", "fin_sector_excluded_from_core")
    ]
    core_candidate = (
        not core_pass and not financial and div_short and len(non_cov_drops) == 0
    )

    fin_drops: list[str] = []
    if financial:
        if div_short:
            fin_drops.append("data_missing:div_history")
        elif non_cut is None or non_cut < NON_CUT_YEARS_REQUIRED:
            fin_drops.append("A_non_cut_years")
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


# ── drop_reason builders ──────────────────────────────────────────────────
def _build_drop_reason_core(row: pd.Series) -> str:
    if row.get("core_pass") is True:
        return "OK"
    reasons: list[str] = []
    cdr = row.get("core_drop_reasons")
    if not _nan(cdr):
        for r in str(cdr).split(";"):
            r = r.strip()
            if r and r.lower() != "nan":
                reasons.append(r)
    # Additional data_missing (skip non_cut — already handled via div_history)
    for col, key in [
        ("effective_yield", "yield"),
        ("cfo_pos_5y_ratio", "cfo_pos_ratio"),
        ("fcf_pos_5y_ratio", "fcf_pos_ratio"),
        ("fcf_payout_3y", "fcf_payout"),
    ]:
        if _nan(row.get(col)):
            tag = f"data_missing:{key}"
            if tag not in reasons:
                reasons.append(tag)
    return "; ".join(reasons) if reasons else "OK"


def _build_drop_reason_satellite(row: pd.Series) -> str:
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


# ── D) Manual check ──────────────────────────────────────────────────────
def _compute_manual_check(row: pd.Series) -> tuple[bool, str]:
    reasons: list[str] = []
    if row.get("high_yield_risk") is True:
        reasons.append("high_yield_risk")
    if row.get("yield_split_mismatch") is True:
        reasons.append("yield_split_mismatch")
    yb = str(row.get("yield_basis", "")).strip()
    if yb == "dps_fallback":
        reasons.append("yield_total_fallback_dps")
    div_src = str(row.get("div_source_used", "")).strip()
    ey = _val(row.get("effective_yield"))
    if div_src == "yf" and ey is not None and ey >= 0.045:
        reasons.append("yf_source_high_yield")
    return len(reasons) > 0, "; ".join(reasons) if reasons else ""


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    h_df = pd.read_csv(HOLDINGS_CSV, dtype=str)
    holdings_codes: set[str] = set(h_df["code_jquants_5digit"].str.strip().dropna())
    assert len(holdings_codes) == 21

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    df["code"] = df["code"].astype(str).str.strip()
    logger.info("Loaded v2_2: %d rows, %d cols", len(df), len(df.columns))

    logger.info("=== BEFORE (v2_2) ===")
    logger.info("  coverage_years median=%.1f", df["coverage_years"].median())
    logger.info("  core_pass=%d", int(df["core_pass"].sum()))
    logger.info("  needs_manual_dividend_check=%d",
                int(df["needs_manual_dividend_check"].sum()))

    # ── A) Re-fetch dividend history via yfinance ─────────────────────────
    refetch_mask = df["coverage_years"] < MIN_COVERAGE_TARGET
    refetch_codes = df.loc[refetch_mask, "code"].tolist()
    logger.info("A) Stocks needing refetch (cov < %d): %d",
                MIN_COVERAGE_TARGET, len(refetch_codes))

    # Holdings first
    refetch_codes = sorted(refetch_codes, key=lambda c: c not in holdings_codes)

    results: dict[str, dict] = {}
    n_ok, n_fail = 0, 0
    t0 = time.time()

    for i, code in enumerate(refetch_codes):
        if i % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(refetch_codes) - i)
            logger.info("  [%d/%d] ok=%d fail=%d  ETA %.0fs",
                        i, len(refetch_codes), n_ok, n_fail, eta)

        r = _fetch_yf_dividends(code)
        if r and r["coverage"] >= 2:
            results[code] = r
            n_ok += 1
        else:
            n_fail += 1

        time.sleep(_YF_DELAY_SEC)

    elapsed = time.time() - t0
    logger.info("A) yfinance done: ok=%d fail=%d in %.0fs", n_ok, n_fail, elapsed)

    # ── Apply new coverage ────────────────────────────────────────────────
    updated = 0
    for idx, row in df.iterrows():
        code = str(row["code"])
        if code not in results:
            continue
        r = results[code]
        old_cov = int(_val(row["coverage_years"]) or 0)
        if r["coverage"] > old_cov:
            new_cov = min(r["coverage"], 10)
            non_cut = _compute_non_cut_years(r["records"])
            df.at[idx, "coverage_years"] = new_cov
            df.at[idx, "non_cut_years"] = non_cut
            df.at[idx, "div_source_used"] = "yf"
            updated += 1
    logger.info("A) Updated coverage for %d stocks", updated)

    # ── B) coverage_short → NaN non_cut_years ─────────────────────────────
    short_mask = df["coverage_years"] < (NON_CUT_YEARS_REQUIRED + 1)
    logger.info("B) coverage_short (< %d): %d stocks",
                NON_CUT_YEARS_REQUIRED + 1, int(short_mask.sum()))

    def _update_flags(flags_str, is_short):
        flags = _parse_flags(flags_str)
        if is_short:
            if "div_history_short" not in flags:
                flags.append("div_history_short")
        else:
            flags = [f for f in flags if f != "div_history_short"]
        return "; ".join(flags) if flags else ""

    df["data_quality_flags"] = [
        _update_flags(f, s) for f, s in zip(df["data_quality_flags"], short_mask)
    ]
    df.loc[short_mask, "non_cut_years"] = np.nan

    # ── C) non_cut_years_verified = bool ──────────────────────────────────
    df["non_cut_years_verified"] = (
        df["coverage_years"] >= (NON_CUT_YEARS_REQUIRED + 1)
    )
    logger.info("C) non_cut_years_verified=True: %d",
                int(df["non_cut_years_verified"].sum()))

    # ── D) needs_manual_dividend_check (yield anomaly only) ───────────────
    df["is_financial"] = df["Sector33CodeName"].apply(_is_financial)
    manual_results = df.apply(_compute_manual_check, axis=1)
    df["needs_manual_dividend_check"] = [r[0] for r in manual_results]
    df["manual_check_reason"] = [r[1] for r in manual_results]
    df["needs_manual_history_check"] = df["data_quality_flags"].apply(
        lambda x: "div_history_short" in str(x)
    )

    # Assert: check↔reason consistency
    chk_true = df[df["needs_manual_dividend_check"] == True]
    bad1 = chk_true[chk_true["manual_check_reason"].apply(
        lambda x: _nan(x) or str(x).strip() == "")]
    assert len(bad1) == 0, f"manual_check=True but reason empty: {len(bad1)}"
    has_reason = df[df["manual_check_reason"].apply(
        lambda x: not _nan(x) and str(x).strip() != "")]
    bad2 = has_reason[has_reason["needs_manual_dividend_check"] != True]
    assert len(bad2) == 0, f"reason non-empty but check=False: {len(bad2)}"
    logger.info("D) Assertions passed ✓")

    # ── Recompute core_pass ───────────────────────────────────────────────
    logger.info("Recomputing core_pass...")
    core_res = df.apply(_recompute_core_pass, axis=1)
    df["core_pass"] = [r[0] for r in core_res]
    df["core_fin_pass"] = [r[1] for r in core_res]
    df["core_candidate"] = [r[2] for r in core_res]
    df["core_drop_reasons"] = [r[3] for r in core_res]

    df["core_momo_pass"] = (
        df["core_pass"] & df["tt_all_pass"].fillna(False).astype(bool)
    )
    df["core_buyable_now"] = (
        df["core_pass"] & (df["trend_score"].fillna(0).astype(int) >= 4)
    )
    df["satellite_buyable_now"] = (
        df["satellite_pass"].fillna(False).astype(bool)
        & df["tt_all_pass"].fillna(False).astype(bool)
    )

    # ── Rebuild drop_reason ───────────────────────────────────────────────
    df["drop_reason_core"] = df.apply(_build_drop_reason_core, axis=1)
    df["drop_reason_satellite"] = df.apply(_build_drop_reason_satellite, axis=1)
    df["drop_reason"] = df.apply(
        lambda row: "OK" if row["core_pass"] or row["satellite_pass"] else
        "; ".join(filter(None, [
            row["drop_reason_core"] if row["drop_reason_core"] != "OK" else "",
            row["drop_reason_satellite"] if row["drop_reason_satellite"] != "OK" else "",
        ])) or "OK",
        axis=1,
    )

    # ── F) Validation logs ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("=== AFTER (v2_3) — VALIDATION ===")
    logger.info("=" * 60)

    cov = df["coverage_years"]
    logger.info("F-1) coverage_years: median=%.1f  mean=%.1f  min=%d  max=%d",
                cov.median(), cov.mean(), int(cov.min()), int(cov.max()))
    logger.info("     distribution:\n%s", cov.value_counts().sort_index().to_string())

    n_core = int(df["core_pass"].sum())
    logger.info("F-2) core_pass = %d  (v2.2 was 1)", n_core)

    n_manual = int(df["needs_manual_dividend_check"].sum())
    logger.info("F-3) needs_manual_dividend_check = %d  (v2.2 was 3016)", n_manual)

    h_mask = df["code"].isin(holdings_codes)
    h_cov6 = int((df.loc[h_mask, "coverage_years"] >= 6).sum())
    logger.info("F-4) Holdings coverage_years>=6: %d / 21", h_cov6)

    logger.info("--- Holdings detail ---")
    for _, row in df[h_mask].sort_values("code").iterrows():
        ey = _val(row.get("effective_yield"))
        logger.info(
            "  %s  cov=%2d  ncut=%4s  verified=%5s  ey=%.4f  core=%s  src=%s  drop=%s",
            row["code"],
            int(row["coverage_years"]),
            str(row.get("non_cut_years", "NaN"))[:4],
            str(row["non_cut_years_verified"])[:5],
            ey if ey else 0,
            row["core_pass"],
            row["div_source_used"],
            str(row.get("drop_reason_core", ""))[:60],
        )

    # ── E) Output ─────────────────────────────────────────────────────────
    out_main = OUT_DIR / "candidates_fixed_v2_3.csv"
    df.to_csv(out_main, index=False, encoding="utf-8-sig")
    logger.info("→ %s  (%d rows, %d cols)", out_main, len(df), len(df.columns))

    # Holdings debug
    debug_cols = [
        "code", "Name", "in_holdings", "is_financial",
        "core_pass", "core_fin_pass", "core_candidate", "core_buyable_now",
        "satellite_pass", "satellite_buyable_now",
        "drop_reason_core", "drop_reason_satellite",
        "yield_basis", "effective_yield",
        "non_cut_years", "non_cut_years_verified",
        "coverage_years", "div_source_used",
        "cfo_pos_5y_ratio", "fcf_pos_5y_ratio", "fcf_payout_3y",
        "value_pass", "eps_fwd", "data_quality_flags",
        "needs_manual_dividend_check", "manual_check_reason",
        "needs_manual_history_check",
    ]
    debug_present = [c for c in debug_cols if c in df.columns]
    h_debug = df[df["in_holdings"] == True][debug_present].copy()
    h_merge = h_df[
        ["code_jquants_5digit", "name_pdf", "account", "shares", "avg_cost_yen"]
    ].merge(h_debug, left_on="code_jquants_5digit", right_on="code", how="left")
    debug_out = OUT_DIR / "holdings_debug_v2_3.csv"
    h_merge.to_csv(debug_out, index=False, encoding="utf-8-sig")
    logger.info("→ %s  (%d rows)", debug_out, len(h_merge))

    # Manual dividend check queue
    queue_cols = [
        "code", "Name", "yield_basis", "effective_yield",
        "non_cut_years_verified", "coverage_years", "fcf_payout_3y",
        "drop_reason_core", "data_quality_flags",
        "manual_check_reason", "div_source_used",
        "high_yield_risk", "yield_split_mismatch",
        "is_financial", "core_pass", "in_holdings",
    ]
    queue_df = df[df["needs_manual_dividend_check"] == True].copy()
    queue_present = [c for c in queue_cols if c in queue_df.columns]
    queue_out = OUT_DIR / "manual_dividend_check_queue_v2_3.csv"
    queue_df[queue_present].to_csv(queue_out, index=False, encoding="utf-8-sig")
    logger.info("→ %s  (%d rows)", queue_out, len(queue_df))

    logger.info("=== patch_v2_3 complete ===")


if __name__ == "__main__":
    main()
