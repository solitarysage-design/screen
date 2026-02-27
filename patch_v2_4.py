#!/usr/bin/env python3
"""patch_v2_4 — Manual check queue triage + data-fill separation.

v2_3 からの変更点:
  1) needs_manual_dividend_check → 3系統に分割
     - needs_manual_shares_check  : yield_basis=dps_fallback
     - needs_manual_dividend_check: high_yield_risk / split_mismatch / yf+high only
     - needs_manual_history_check : div_history_short (v2_3維持)
  2) yield_basis=none → needs_data_fill (監査と欠損を分離)
  3) effective_yield / high_yield_risk の一貫性

パイプライン:
  A) yfinance 配当履歴拡張 (v2_3同等, --skip-yf でスキップ可)
  B) coverage_short → div_history_short フラグ管理
  C) non_cut_years_verified
  D) effective_yield / high_yield_risk 再計算
  E) 3-way manual check split + data_fill
  F) Core pass 再計算
  G) Output
  H) Validation
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
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
import os as _os
PROJECT_DIR = Path(_os.environ.get("SCREEN_PROJECT_DIR", str(Path(__file__).resolve().parent)))
OUTPUT_DIR = Path(_os.environ.get("SCREEN_OUTPUT_DIR", str(PROJECT_DIR / "output" / date.today().strftime("%Y%m%d"))))
HOLDINGS_CSV = Path(_os.environ.get("SCREEN_HOLDINGS_CSV", str(Path.home() / "Downloads" / "holdings_extracted_20260224.csv")))
INPUT_CSV = OUTPUT_DIR / "candidates_fixed_v2_2.csv"
OUT_DIR = OUTPUT_DIR

# ── Constants ─────────────────────────────────────────────────────────────────
NON_CUT_YEARS_REQUIRED = 5
CORE_YIELD_MIN = 0.03
HIGH_YIELD_RISK_THRESHOLD = 0.06
YF_HIGH_YIELD_THRESHOLD = 0.045
MIN_COVERAGE_TARGET = 6

FINANCIAL_EXACT = {
    "銀行業", "保険業", "その他金融業",
    "証券、商品先物取引業", "証券･商品先物取引業",
}
FINANCIAL_SUBSTRINGS = ("証券", "商品先物", "銀行", "保険")

_CUT_TOL_ABS_JPY = 1.0
_CUT_TOL_REL = 0.02
_NON_CUT_MAX_Y = 10
_YF_DELAY_SEC = 0.25


# ── Helpers ───────────────────────────────────────────────────────────────────
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


# ── A) yfinance dividend history fetcher (from v2_3) ─────────────────────────
def _fetch_yf_dividends(code5: str, max_retries: int = 2) -> dict | None:
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

        try:
            annual = adj_divs.resample("YE").sum()
        except Exception:
            annual = adj_divs.resample("A").sum()
        annual = annual[annual > 0]
        if annual.empty:
            return None

        records = [(int(dt.year), float(dps)) for dt, dps in annual.items()]
        records = sorted(records, key=lambda x: x[0], reverse=True)[:10]

        return {
            "records": records,
            "coverage": len(records),
            "has_splits": has_splits,
            "split_adjusted": split_adjusted,
        }
    except Exception as exc:
        logger.debug("yfinance %s: %s", code5, str(exc)[:80])
        return None


# ── Dividend-cut detection ────────────────────────────────────────────────────
def _is_div_cut(curr: float, prev: float) -> bool:
    if prev <= 0:
        return False
    diff = prev - curr
    if diff <= 0:
        return False
    return diff > _CUT_TOL_ABS_JPY and diff / prev > _CUT_TOL_REL


def _compute_non_cut_years(records: list[tuple[int, float]]) -> float | None:
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


# ── Core pass recompute ──────────────────────────────────────────────────────
def _recompute_core_pass(row: pd.Series) -> tuple[bool, bool, bool, str]:
    from screen.config import CYCLICAL_SECTOR33

    financial = bool(row.get("is_financial", False))
    sector = str(row.get("Sector33CodeName") or "")
    cyclical = sector in CYCLICAL_SECTOR33
    payout_cap = 0.60 if cyclical else 0.70

    non_cut = _val(row.get("non_cut_years"))
    div_short = "div_history_short" in str(row.get("data_quality_flags", ""))

    fund_drops: list[str] = []

    if div_short:
        fund_drops.append("data_missing:div_history")
    elif non_cut is None or non_cut < NON_CUT_YEARS_REQUIRED:
        fund_drops.append("A_non_cut_years")

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

    ey = _val(row.get("effective_yield"))
    if ey is None or ey < CORE_YIELD_MIN:
        fund_drops.append("E_div_yield")

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


# ── Drop-reason builders ─────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# v2_4 NEW: 3-way manual check split + data_fill
# ══════════════════════════════════════════════════════════════════════════════

def _compute_shares_check(row: pd.Series) -> tuple[bool, str]:
    """1-1) needs_manual_shares_check: DPS*shares fallback used."""
    yb = str(row.get("yield_basis", "")).strip()
    # yield_basis=none → data_fill, not shares check
    if yb == "none":
        return False, ""

    reasons: list[str] = []
    if yb == "dps_fallback":
        reasons.append("yield_total_fallback_dps")
    flags = _parse_flags(row.get("data_quality_flags", ""))
    if "yield_total_fallback_dps" in flags and "yield_total_fallback_dps" not in reasons:
        reasons.append("yield_total_fallback_dps")

    return len(reasons) > 0, "; ".join(reasons)


def _compute_dividend_check_v24(row: pd.Series) -> tuple[bool, str]:
    """1-2) needs_manual_dividend_check: truly suspicious yield only."""
    yb = str(row.get("yield_basis", "")).strip()
    # yield_basis=none → data_fill, not dividend check
    if yb == "none":
        return False, ""

    reasons: list[str] = []
    ey = _val(row.get("effective_yield"))

    # high_yield_risk (recalculated in step D)
    if row.get("high_yield_risk") is True:
        reasons.append("high_yield_risk")

    # yield_split_mismatch
    if row.get("yield_split_mismatch") is True:
        reasons.append("yield_split_mismatch")

    # yf source + moderately high yield
    div_src = str(row.get("div_source_used", "")).strip()
    if div_src == "yf" and ey is not None and ey >= YF_HIGH_YIELD_THRESHOLD:
        reasons.append("yf_source_high_yield")

    return len(reasons) > 0, "; ".join(reasons)


def _compute_history_check(row: pd.Series) -> tuple[bool, str]:
    """1-3) needs_manual_history_check: div_history_short (v2_3 維持)."""
    flags = _parse_flags(row.get("data_quality_flags", ""))
    if "div_history_short" not in flags:
        return False, ""
    cov = int(_val(row.get("coverage_years")) or 0)
    reason = f"div_history_short;coverage_years={cov}"
    return True, reason


def _compute_data_fill(row: pd.Series) -> tuple[bool, str]:
    """2) needs_data_fill: yield_basis=none → data gap, not audit."""
    yb = str(row.get("yield_basis", "")).strip()
    if yb != "none":
        return False, ""

    missing: list[str] = []
    if _nan(row.get("dps_fwd")):
        missing.append("missing_dps")
    if _nan(row.get("total_div_fwd")) and _nan(row.get("total_div_actual")):
        missing.append("missing_total_div")
    if _nan(row.get("net_shares_latest")):
        missing.append("missing_shares")
    if _nan(row.get("price")):
        missing.append("missing_price")

    if not missing:
        missing.append("yield_calc_failed")

    return True, "; ".join(missing)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="patch_v2_4")
    parser.add_argument(
        "--skip-yf", action="store_true",
        help="Skip yfinance dividend history fetching (use existing coverage)",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Override input CSV path (default: v2_2)",
    )
    args = parser.parse_args()

    input_csv = Path(args.input) if args.input else INPUT_CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    h_df = pd.read_csv(HOLDINGS_CSV, dtype=str)
    holdings_codes: set[str] = set(h_df["code_jquants_5digit"].str.strip().dropna())
    assert len(holdings_codes) == 21

    df = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)
    df["code"] = df["code"].astype(str).str.strip()
    logger.info("Loaded %s: %d rows, %d cols", input_csv.name, len(df), len(df.columns))

    logger.info("=== BEFORE ===")
    logger.info("  yield_basis: %s", df["yield_basis"].value_counts().to_dict())
    logger.info("  needs_manual_dividend_check: %d",
                int(df["needs_manual_dividend_check"].sum()))
    logger.info("  coverage_years median=%.1f", df["coverage_years"].median())

    # ══════════════════════════════════════════════════════════════════════════
    # A) yfinance dividend history expansion (v2_3 logic)
    # ══════════════════════════════════════════════════════════════════════════
    if args.skip_yf:
        logger.info("A) --skip-yf: skipping yfinance fetch")
    else:
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

        # Apply new coverage
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

    # ══════════════════════════════════════════════════════════════════════════
    # B) coverage_short → div_history_short flag management
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # C) non_cut_years_verified
    # ══════════════════════════════════════════════════════════════════════════
    df["non_cut_years_verified"] = (
        df["coverage_years"] >= (NON_CUT_YEARS_REQUIRED + 1)
    )
    logger.info("C) non_cut_years_verified=True: %d",
                int(df["non_cut_years_verified"].sum()))

    # ══════════════════════════════════════════════════════════════════════════
    # D) effective_yield / high_yield_risk recalculation
    # ══════════════════════════════════════════════════════════════════════════
    # effective_yield priority: fwd → actual → dps_fallback, none→NaN
    df["effective_yield"] = np.nan
    fwd_mask = df["yield_basis"] == "fwd"
    df.loc[fwd_mask, "effective_yield"] = df.loc[fwd_mask, "dividend_yield_total_fwd"]
    actual_mask = df["yield_basis"] == "actual"
    df.loc[actual_mask, "effective_yield"] = df.loc[actual_mask, "dividend_yield_total_actual"]
    fb_mask = df["yield_basis"] == "dps_fallback"
    df.loc[fb_mask, "effective_yield"] = df.loc[fb_mask, "dividend_yield_total_fwd"]
    # yield_basis=="none" → effective_yield stays NaN

    # high_yield_risk: uniform threshold on effective_yield
    df["high_yield_risk"] = df["effective_yield"].fillna(0) >= HIGH_YIELD_RISK_THRESHOLD
    df["is_financial"] = df["Sector33CodeName"].apply(_is_financial)

    logger.info("D) effective_yield non-NaN: %d, high_yield_risk: %d",
                int(df["effective_yield"].notna().sum()),
                int(df["high_yield_risk"].sum()))

    # ══════════════════════════════════════════════════════════════════════════
    # E) 3-way manual check split + data_fill  (v2_4 core change)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("E) Computing 3-way manual check split...")

    # E-1) shares check
    sc_results = df.apply(_compute_shares_check, axis=1)
    df["needs_manual_shares_check"] = [r[0] for r in sc_results]
    df["shares_check_reason"] = [r[1] for r in sc_results]

    # E-2) dividend check (v2_4 tightened)
    dc_results = df.apply(_compute_dividend_check_v24, axis=1)
    df["needs_manual_dividend_check"] = [r[0] for r in dc_results]
    df["manual_dividend_check_reason"] = [r[1] for r in dc_results]

    # E-3) history check
    hc_results = df.apply(_compute_history_check, axis=1)
    df["needs_manual_history_check"] = [r[0] for r in hc_results]
    df["history_check_reason"] = [r[1] for r in hc_results]

    # E-4) data fill (yield_basis=none)
    df_results = df.apply(_compute_data_fill, axis=1)
    df["needs_data_fill"] = [r[0] for r in df_results]
    df["data_fill_reason"] = [r[1] for r in df_results]

    # ── Assertions: flag=True ↔ reason non-empty ─────────────────────────────
    for flag_col, reason_col in [
        ("needs_manual_shares_check", "shares_check_reason"),
        ("needs_manual_dividend_check", "manual_dividend_check_reason"),
        ("needs_manual_history_check", "history_check_reason"),
        ("needs_data_fill", "data_fill_reason"),
    ]:
        flagged = df[df[flag_col] == True]
        empty_reason = flagged[flagged[reason_col].apply(
            lambda x: _nan(x) or str(x).strip() == "")]
        assert len(empty_reason) == 0, (
            f"{flag_col}=True but {reason_col} empty: {len(empty_reason)} rows"
        )
        has_reason = df[df[reason_col].apply(
            lambda x: not _nan(x) and str(x).strip() != "")]
        bad = has_reason[has_reason[flag_col] != True]
        assert len(bad) == 0, (
            f"{reason_col} non-empty but {flag_col}=False: {len(bad)} rows"
        )
    logger.info("E) All flag↔reason assertions passed")

    # Remove old v2_3 manual_check_reason column if present
    if "manual_check_reason" in df.columns:
        df.drop(columns=["manual_check_reason"], inplace=True)

    # ══════════════════════════════════════════════════════════════════════════
    # F) Core pass recompute
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("F) Recomputing core_pass...")
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

    # Rebuild drop_reason
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

    logger.info("F) core_pass=%d, core_fin_pass=%d, core_candidate=%d",
                int(df["core_pass"].sum()),
                int(df["core_fin_pass"].sum()),
                int(df["core_candidate"].sum()))

    # ══════════════════════════════════════════════════════════════════════════
    # G) Output
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("G) Writing output files...")

    # G-1) Main CSV
    out_main = OUT_DIR / "candidates_fixed_v2_4.csv"
    df.to_csv(out_main, index=False, encoding="utf-8-sig")
    logger.info("  → %s  (%d rows, %d cols)", out_main.name, len(df), len(df.columns))

    # G-2) Manual shares check queue
    sc_cols = [
        "code", "Name", "yield_basis", "effective_yield",
        "data_quality_flags", "shares_check_reason",
    ]
    sc_df = df[df["needs_manual_shares_check"] == True].copy()
    sc_present = [c for c in sc_cols if c in sc_df.columns]
    sc_out = OUT_DIR / "manual_shares_check_queue_v2_4.csv"
    sc_df[sc_present].to_csv(sc_out, index=False, encoding="utf-8-sig")
    logger.info("  → %s  (%d rows)", sc_out.name, len(sc_df))

    # G-3) Manual dividend check queue
    dc_cols = [
        "code", "Name", "yield_basis", "effective_yield",
        "manual_dividend_check_reason",
    ]
    dc_df = df[df["needs_manual_dividend_check"] == True].copy()
    dc_present = [c for c in dc_cols if c in dc_df.columns]
    dc_out = OUT_DIR / "manual_dividend_check_queue_v2_4.csv"
    dc_df[dc_present].to_csv(dc_out, index=False, encoding="utf-8-sig")
    logger.info("  → %s  (%d rows)", dc_out.name, len(dc_df))

    # G-4) Data fill queue
    fill_cols = [
        "code", "Name", "yield_basis", "data_fill_reason",
    ]
    fill_df = df[df["needs_data_fill"] == True].copy()
    fill_present = [c for c in fill_cols if c in fill_df.columns]
    fill_out = OUT_DIR / "data_fill_queue_v2_4.csv"
    fill_df[fill_present].to_csv(fill_out, index=False, encoding="utf-8-sig")
    logger.info("  → %s  (%d rows)", fill_out.name, len(fill_df))

    # G-5) Holdings debug
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
        "needs_manual_shares_check", "shares_check_reason",
        "needs_manual_dividend_check", "manual_dividend_check_reason",
        "needs_manual_history_check", "history_check_reason",
        "needs_data_fill", "data_fill_reason",
        "high_yield_risk", "yield_split_mismatch",
    ]
    debug_present = [c for c in debug_cols if c in df.columns]
    h_debug = df[df["in_holdings"] == True][debug_present].copy()
    h_merge = h_df[
        ["code_jquants_5digit", "name_pdf", "account", "shares", "avg_cost_yen"]
    ].merge(h_debug, left_on="code_jquants_5digit", right_on="code", how="left")
    debug_out = OUT_DIR / "holdings_debug_v2_4.csv"
    h_merge.to_csv(debug_out, index=False, encoding="utf-8-sig")
    logger.info("  → %s  (%d rows)", debug_out.name, len(h_merge))

    # ══════════════════════════════════════════════════════════════════════════
    # H) Validation logs
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("=== v2_4 VALIDATION ===")
    logger.info("=" * 60)

    n_shares_chk = int(df["needs_manual_shares_check"].sum())
    n_div_chk = int(df["needs_manual_dividend_check"].sum())
    n_hist_chk = int(df["needs_manual_history_check"].sum())
    n_data_fill = int(df["needs_data_fill"].sum())
    n_yield_none = int((df["yield_basis"] == "none").sum())

    logger.info("H-1) Queue sizes:")
    logger.info("  needs_manual_shares_check   = %d", n_shares_chk)
    logger.info("  needs_manual_dividend_check  = %d  (was ~1787 in v2_3)", n_div_chk)
    logger.info("  needs_manual_history_check   = %d", n_hist_chk)
    logger.info("  needs_data_fill              = %d", n_data_fill)

    logger.info("H-2) Consistency checks:")
    logger.info("  yield_basis=none count       = %d", n_yield_none)
    logger.info("  needs_data_fill count         = %d  (should == yield_basis=none)",
                n_data_fill)
    assert n_data_fill == n_yield_none, (
        f"needs_data_fill ({n_data_fill}) != yield_basis=none ({n_yield_none})"
    )

    # No yield_basis=none in shares/dividend check
    none_in_shares = int(
        (df["needs_manual_shares_check"] & (df["yield_basis"] == "none")).sum()
    )
    none_in_div = int(
        (df["needs_manual_dividend_check"] & (df["yield_basis"] == "none")).sum()
    )
    logger.info("  yield_basis=none in shares_check = %d  (should be 0)", none_in_shares)
    logger.info("  yield_basis=none in div_check    = %d  (should be 0)", none_in_div)
    assert none_in_shares == 0, f"yield_basis=none in shares_check: {none_in_shares}"
    assert none_in_div == 0, f"yield_basis=none in div_check: {none_in_div}"

    # Reason non-empty check (already asserted above, log for clarity)
    for flag_col, reason_col in [
        ("needs_manual_shares_check", "shares_check_reason"),
        ("needs_manual_dividend_check", "manual_dividend_check_reason"),
        ("needs_manual_history_check", "history_check_reason"),
        ("needs_data_fill", "data_fill_reason"),
    ]:
        flagged = df[df[flag_col] == True]
        empty = flagged[flagged[reason_col].apply(
            lambda x: _nan(x) or str(x).strip() == "")]
        logger.info("  %s=True with empty reason: %d  (must be 0)",
                     flag_col, len(empty))

    logger.info("H-3) yield_basis breakdown:")
    logger.info("%s", df["yield_basis"].value_counts().to_string())

    logger.info("H-4) high_yield_risk: %d", int(df["high_yield_risk"].sum()))

    logger.info("H-5) core_pass=%d, satellite_pass=%d",
                int(df["core_pass"].sum()),
                int(df.get("satellite_pass", pd.Series(dtype=bool)).fillna(False).sum()))

    # Holdings detail
    h_mask = df["code"].isin(holdings_codes)
    logger.info("--- Holdings detail ---")
    for _, row in df[h_mask].sort_values("code").iterrows():
        ey = _val(row.get("effective_yield"))
        queues = []
        if row.get("needs_manual_shares_check"):
            queues.append("SC")
        if row.get("needs_manual_dividend_check"):
            queues.append("DC")
        if row.get("needs_manual_history_check"):
            queues.append("HC")
        if row.get("needs_data_fill"):
            queues.append("DF")
        logger.info(
            "  %s  cov=%2d  ncut=%4s  ey=%.4f  core=%5s  yb=%-13s  queues=[%s]",
            row["code"],
            int(row["coverage_years"]),
            str(row.get("non_cut_years", "NaN"))[:4],
            ey if ey else 0.0,
            str(row["core_pass"])[:5],
            str(row["yield_basis"]),
            ",".join(queues) if queues else "none",
        )

    logger.info("=" * 60)
    logger.info("=== patch_v2_4 complete ===")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
