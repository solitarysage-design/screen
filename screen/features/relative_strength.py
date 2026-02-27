"""Relative Strength score computation."""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from screen.config import RS_WEIGHTS

logger = logging.getLogger(__name__)

# Approximate trading days for each window
_WINDOW_DAYS: dict[str, int] = {
    "3m": 63,
    "6m": 126,
    "9m": 189,
    "12m": 252,
}


def _period_return(close: pd.Series, days: int) -> float:
    """Compute return over last *days* observations."""
    if len(close) < days + 1:
        return np.nan
    try:
        arr = close.values
        end = float(arr[-1])
        start = float(arr[-(days + 1)])
    except (TypeError, ValueError, IndexError):
        return np.nan
    if start == 0 or np.isnan(start) or np.isnan(end):
        return np.nan
    return (end / start) - 1.0


def compute_rs(
    prices: pd.DataFrame,
    topix: pd.DataFrame | None,
    mode: str = "universe_percentile",
    asof: date | None = None,
    weights: dict[str, float] = RS_WEIGHTS,
) -> pd.DataFrame:
    """Compute RS score and percentile.

    Args:
        prices: DataFrame [Date, Code, Close, Volume].
        topix: TOPIX prices DataFrame [Date, Close] or None.
        mode: "topix" | "universe_percentile"
        asof: Snapshot date (defaults to max date in prices).
        weights: Period weights.

    Returns:
        DataFrame [code, rs_score, rs_percentile]
    """
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])

    if asof is not None:
        prices = prices[prices["Date"] <= pd.Timestamp(asof)]

    # TOPIX per-period returns
    topix_returns: dict[str, float] = {}
    if mode == "topix" and topix is not None:
        topix = topix.copy()
        topix["Date"] = pd.to_datetime(topix["Date"])
        if asof is not None:
            topix = topix[topix["Date"] <= pd.Timestamp(asof)]
        topix = topix.sort_values("Date").drop_duplicates("Date")
        topix_close = topix["Close"].astype(float)
        for period, days in _WINDOW_DAYS.items():
            topix_returns[period] = _period_return(topix_close, days)

    rows = []
    for code, grp in prices.groupby("Code"):
        grp = grp.sort_values("Date").drop_duplicates(subset="Date", keep="last")
        close = grp["Close"].astype(float).reset_index(drop=True)

        period_scores: dict[str, float] = {}
        for period, days in _WINDOW_DAYS.items():
            ret = _period_return(close, days)
            if np.isnan(ret):
                period_scores[period] = np.nan
                continue

            if mode == "topix" and period in topix_returns and not np.isnan(topix_returns[period]):
                # Relative return vs TOPIX
                period_scores[period] = ret - topix_returns[period]
            else:
                # Raw return (will be percentile-ranked across universe later)
                period_scores[period] = ret

        # Weighted score
        total_weight = 0.0
        weighted_sum = 0.0
        for period, w in weights.items():
            s = period_scores.get(period, np.nan)
            if not np.isnan(s):
                weighted_sum += s * w
                total_weight += w

        rs_score = weighted_sum / total_weight if total_weight > 0 else np.nan

        rows.append({
            "code": str(code),
            "rs_score": rs_score,
            "ret_3m": period_scores.get("3m", np.nan),
            "ret_6m": period_scores.get("6m", np.nan),
            "ret_9m": period_scores.get("9m", np.nan),
            "ret_12m": period_scores.get("12m", np.nan),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Percentile rank (0-100, higher = stronger)
    df["rs_percentile"] = df["rs_score"].rank(pct=True, na_option="keep") * 100

    return df[["code", "rs_score", "rs_percentile", "ret_3m", "ret_6m", "ret_9m", "ret_12m"]]
