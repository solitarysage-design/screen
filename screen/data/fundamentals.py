"""Fundamental data retrieval using J-Quants fins/statements.

Unit contract: ALL monetary values stored as TOTAL YEN (整数 or float).
DPS (per-share) is kept separately for yield calculation only.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from screen.data.jquants_client import fetch_statements, get_client
from screen.data.prices import _jq_to_yf_ticker

logger = logging.getLogger(__name__)

# Columns we need from fins/statements
_STMT_NUM_COLS = [
    "CashFlowsFromOperatingActivities",
    "CashFlowsFromInvestingActivities",
    "ResultTotalDividendPaidAnnual",
    "ForecastTotalDividendPaidAnnual",
    "ResultDividendPerShareAnnual",
    "ForecastDividendPerShareAnnual",
    "EarningsPerShare",
    "ForecastEarningsPerShare",
    "BookValuePerShare",
    "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
    "NumberOfTreasuryStockAtTheEndOfFiscalYear",
]
_FY_PERIOD_VALUES = {"FY"}


def _to_float(val) -> float | None:
    """Convert string/numeric to float. Empty string or None → None."""
    if val is None:
        return None
    if isinstance(val, float):
        return None if np.isnan(val) else val
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fetch_stmt_df(code: str) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch fins/statements and convert numeric columns to float dtype.

    Returns:
        (df_or_None, error_tag_or_None)
        error_tag is set when the fetch failed (e.g. "stmt_fetch_failed:no_auth").
    """
    import time as _time
    last_exc: Exception | None = None
    for attempt, wait_s in enumerate([0, 1, 2, 4, 8]):
        if wait_s > 0:
            _time.sleep(wait_s)
        try:
            raw = fetch_statements(code=code)
            if isinstance(raw, pd.DataFrame):
                df = raw.copy()
            elif isinstance(raw, list):
                df = pd.DataFrame(raw)
            else:
                return None, None
            if df.empty:
                return None, None
            for col in _STMT_NUM_COLS:
                if col in df.columns:
                    df[col] = df[col].apply(_to_float)
            return df, None
        except Exception as exc:
            exc_str = str(exc)
            # Auth errors are permanent — don't retry
            if "mail_address" in exc_str or "refresh_token" in exc_str or "password" in exc_str:
                logger.warning("%s: fins/statements auth failed (attempt %d): %s", code, attempt + 1, exc_str[:80])
                return None, f"stmt_fetch_failed:no_jquants_auth"
            last_exc = exc
            logger.warning("%s: fins/statements fetch failed (attempt %d/%d): %s", code, attempt + 1, 5, exc_str[:80])

    err_tag = f"stmt_fetch_failed:{type(last_exc).__name__}" if last_exc else "stmt_fetch_failed:unknown"
    return None, err_tag


def _extract_fy_records(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return up to n most-recent FY annual records, sorted descending by FY end date."""
    fy = df[df["TypeOfCurrentPeriod"].isin(_FY_PERIOD_VALUES)].copy()
    if fy.empty:
        return fy
    date_col = "CurrentFiscalYearEndDate"
    if date_col in fy.columns:
        fy[date_col] = pd.to_datetime(fy[date_col], errors="coerce")
        fy = fy.sort_values(date_col, ascending=False)
    return fy.head(n)


def _extract_quarter_eps(df: pd.DataFrame, max_q: int = 8) -> list[float | None]:
    """Return list of quarterly EPS (most-recent-first) for O'Neil scoring.

    We use cumulative-from-FY-start EPS for each quarterly report.
    For YoY comparisons, caller compares same quarter across fiscal years.
    """
    q_df = df[df["TypeOfCurrentPeriod"].isin({"1Q", "2Q", "3Q"})].copy()
    fy_df = df[df["TypeOfCurrentPeriod"].isin(_FY_PERIOD_VALUES)].copy()

    combined = pd.concat([q_df, fy_df], ignore_index=True)
    if combined.empty:
        return []

    date_col = "CurrentFiscalYearEndDate"
    period_col = "TypeOfCurrentPeriod"
    period_order = {"3Q": 3, "2Q": 2, "1Q": 1, "FY": 4}

    if date_col in combined.columns:
        combined[date_col] = pd.to_datetime(combined[date_col], errors="coerce")

    if date_col in combined.columns and period_col in combined.columns:
        combined["_period_num"] = combined[period_col].map(period_order).fillna(0)
        combined = combined.sort_values(
            [date_col, "_period_num"], ascending=[False, False]
        )

    eps_vals = []
    for _, row in combined.iterrows():
        v = _to_float(row.get("EarningsPerShare"))
        eps_vals.append(v)
        if len(eps_vals) >= max_q:
            break
    return eps_vals


def get_fundamentals(codes: list[str]) -> pd.DataFrame:
    """Fetch fundamentals for all codes using J-Quants fins/statements.

    Returns one row per code with columns:
        code, cfo_annual, cfi_annual, fcf_annual, div_paid_annual,
        dps_actual_annual, dps_fwd, bps_latest, eps_fwd,
        net_shares_latest, eps_q_list,
        data_coverage, missing_fields, data_quality_flags, error_flags
    """
    rows = []

    for code in codes:
        df, err_tag = _fetch_stmt_df(code)
        row = _build_row(code, df, err_tag)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_row(code: str, df: pd.DataFrame | None, err_tag: str | None = None) -> dict[str, Any]:
    quality_flags: list[str] = []
    error_flags: list[str] = []
    missing: list[str] = []

    if err_tag:
        error_flags.append(err_tag)

    if df is None or df.empty:
        return {
            "code": code,
            "cfo_annual": None, "cfi_annual": None, "fcf_annual": None,
            "div_paid_annual": None, "dps_actual_annual": None,
            "shares_annual": None, "jq_fy_dates": [],
            "total_div_fwd": None, "total_div_actual": None,
            "dps_fwd": None, "bps_latest": None, "eps_fwd": None,
            "net_shares_latest": None, "eps_q_list": None,
            "data_coverage": 0.0, "missing_fields": ["all"],
            "data_quality_flags": ["no_stmt_data"],
            "error_flags": error_flags,
        }

    fy = _extract_fy_records(df)

    # ── FY end dates (most-recent-first) for dividend history source ──────
    jq_fy_dates: list[str] = []
    if "CurrentFiscalYearEndDate" in fy.columns:
        jq_fy_dates = [
            str(v.date()) if pd.notna(v) else ""
            for v in fy["CurrentFiscalYearEndDate"].tolist()
        ]

    # ── CFO / CFI / FCF (annual list, total yen) ──────────────────────────
    cfo_list = _col_list(fy, "CashFlowsFromOperatingActivities")
    cfi_list = _col_list(fy, "CashFlowsFromInvestingActivities")
    fcf_list = _compute_fcf(cfo_list, cfi_list)

    # ── Dividends paid (total yen, primary from ResultTotalDividendPaid) ──
    div_paid_list = _col_list(fy, "ResultTotalDividendPaidAnnual")

    # ── Fill missing div_paid via DPS × net_shares ───────────────────────
    dps_list = _col_list(fy, "ResultDividendPerShareAnnual")
    issued_list = _col_list(fy, "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock")
    treasury_list = _col_list(fy, "NumberOfTreasuryStockAtTheEndOfFiscalYear")
    net_shares_list = _compute_net_shares(issued_list, treasury_list)

    div_paid_list = _fill_div_paid(div_paid_list, dps_list, net_shares_list)

    # Check unit sanity for div_paid (should be billions of yen range for large caps)
    _check_unit_sanity(div_paid_list, "div_paid", quality_flags, lo=1e6, hi=1e16)
    _check_unit_sanity(cfo_list, "cfo", quality_flags, lo=1e6, hi=1e15)

    # ── Forward / latest point-in-time values ────────────────────────────
    latest_fy = fy.iloc[0] if not fy.empty else None

    dps_fwd = _first_valid([
        _to_float(latest_fy.get("ForecastDividendPerShareAnnual")) if latest_fy is not None else None,
        _to_float(latest_fy.get("ResultDividendPerShareAnnual")) if latest_fy is not None else None,
    ])

    bps_latest = _to_float(latest_fy.get("BookValuePerShare")) if latest_fy is not None else None

    eps_fwd = _to_float(latest_fy.get("ForecastEarningsPerShare")) if latest_fy is not None else None
    if eps_fwd is None:
        eps_fwd = _to_float(latest_fy.get("EarningsPerShare")) if latest_fy is not None else None

    net_shares_latest = net_shares_list[0] if net_shares_list else None
    if net_shares_latest is None:
        quality_flags.append("shares_missing")

    # ── Total dividend (split-invariant): Forecast 優先、なければ Result ──
    total_div_fwd = _to_float(
        latest_fy.get("ForecastTotalDividendPaidAnnual")
    ) if latest_fy is not None else None
    total_div_actual = _to_float(
        latest_fy.get("ResultTotalDividendPaidAnnual")
    ) if latest_fy is not None else None
    # ResultTotalDividendPaidAnnual が無い場合は div_paid_list[0] で補完
    if total_div_actual is None and div_paid_list:
        total_div_actual = div_paid_list[0]

    # ── Quarterly EPS for O'Neil ─────────────────────────────────────────
    eps_q_list = _extract_quarter_eps(df)

    # ── Missing field tracking ────────────────────────────────────────────
    check_map = {
        "cfo": cfo_list, "cfi": cfi_list, "div_paid": div_paid_list,
        "dps_fwd": [dps_fwd], "bps": [bps_latest], "eps_q": eps_q_list,
    }
    for name, lst in check_map.items():
        if not lst or all(v is None for v in lst):
            missing.append(name)

    all_fields = ["cfo", "cfi", "div_paid", "dps_fwd", "bps", "eps_q"]
    coverage = round((len(all_fields) - len(missing)) / len(all_fields), 2)

    logger.debug(
        "%s: FY=%d cfo[0]=%s cfi[0]=%s div_paid[0]=%s dps_fwd=%s bps=%s",
        code, len(fy),
        _fmt(cfo_list, 0), _fmt(cfi_list, 0), _fmt(div_paid_list, 0),
        dps_fwd, bps_latest,
    )

    return {
        "code": code,
        "cfo_annual": cfo_list or None,
        "cfi_annual": cfi_list or None,
        "fcf_annual": fcf_list or None,
        "div_paid_annual": div_paid_list or None,
        "dps_actual_annual": dps_list or None,
        "shares_annual": net_shares_list or None,
        "jq_fy_dates": jq_fy_dates,
        "total_div_fwd": total_div_fwd,
        "total_div_actual": total_div_actual,
        "dps_fwd": dps_fwd,
        "bps_latest": bps_latest,
        "eps_fwd": eps_fwd,
        "net_shares_latest": net_shares_latest,
        "eps_q_list": eps_q_list or None,
        "data_coverage": coverage,
        "missing_fields": missing,
        "data_quality_flags": quality_flags,
        "error_flags": error_flags,
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _col_list(df: pd.DataFrame, col: str) -> list[float | None]:
    if col not in df.columns or df.empty:
        return []
    return [_to_float(v) for v in df[col].tolist()]


def _compute_fcf(cfo: list, cfi: list) -> list[float | None]:
    n = min(len(cfo), len(cfi))
    result = []
    for i in range(n):
        c, i_ = cfo[i], cfi[i]
        if c is None or i_ is None:
            result.append(None)
        else:
            result.append(c + i_)  # CFI is negative, so this is CFO - |CFI|
    return result


def _compute_net_shares(issued: list, treasury: list) -> list[float | None]:
    n = min(len(issued), len(treasury))
    result = []
    for i in range(n):
        a, b = issued[i], treasury[i]
        if a is None:
            result.append(None)
        else:
            result.append(a - (b or 0.0))
    return result


def _fill_div_paid(
    div_paid: list, dps: list, net_shares: list
) -> list[float | None]:
    """Fill None in div_paid using DPS x net_shares fallback."""
    n = max(len(div_paid), len(dps))
    result = []
    for i in range(n):
        dp = div_paid[i] if i < len(div_paid) else None
        if dp is not None and dp > 0:
            result.append(dp)
        else:
            d = dps[i] if i < len(dps) else None
            ns = net_shares[i] if i < len(net_shares) else None
            if d is not None and ns is not None and ns > 0:
                result.append(d * ns)
            else:
                result.append(None)
    return result


def _check_unit_sanity(
    lst: list, name: str, flags: list[str], lo: float, hi: float
) -> None:
    vals = [v for v in lst if v is not None]
    if not vals:
        return
    nonzero = [abs(v) for v in vals if v != 0]
    if not nonzero:
        return
    m = max(nonzero)
    if m < lo:
        flags.append(f"unit_too_small:{name}:{m:.2e}")
    if m > hi:
        flags.append(f"unit_too_large:{name}:{m:.2e}")


def _first_valid(lst: list) -> Any:
    for v in lst:
        if v is not None:
            return v
    return None


def _fmt(lst: list, idx: int) -> str:
    if not lst or idx >= len(lst) or lst[idx] is None:
        return "None"
    return f"{lst[idx]:,.0f}"
