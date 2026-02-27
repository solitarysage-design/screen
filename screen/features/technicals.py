"""Technical indicators: SMA, 52-week high/low."""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from screen.config import SMA_PERIODS

logger = logging.getLogger(__name__)

_52W_DAYS = 252  # rolling window for 52-week high/low


def compute_technicals(prices: pd.DataFrame, asof: date) -> pd.DataFrame:
    """Compute SMA50/150/200 and 52-week high/low as of *asof* date.

    Args:
        prices: DataFrame with columns [Date, Code, Close, Volume].
        asof: Snapshot date (uses data up to and including this date).

    Returns:
        DataFrame with one row per code:
            code, price, sma50, sma150, sma200,
            high52w, low52w, sma200_20d_ago
    """
    asof_ts = pd.Timestamp(asof)
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])

    # Restrict to data up to asof
    prices = prices[prices["Date"] <= asof_ts]

    rows = []
    for code, grp in prices.groupby("Code"):
        grp = grp.sort_values("Date").drop_duplicates(subset="Date", keep="last")
        close = grp["Close"].astype(float).reset_index(drop=True)

        if close.empty:
            logger.warning("%s: no price data up to %s, skipping", code, asof)
            continue

        current_price = float(close.values[-1])

        # SMAs
        smas = {}
        sma200_arr = None
        for period in SMA_PERIODS:
            if len(close) >= period:
                sma_arr = close.rolling(period).mean().values
                sma_val = float(sma_arr[-1])
                if period == 200:
                    sma200_arr = sma_arr
            else:
                sma_val = np.nan
            smas[f"sma{period}"] = sma_val

        # SMA200 value 20 trading days ago
        sma200_20d_ago = np.nan
        if sma200_arr is not None and len(sma200_arr) > 20:
            sma200_20d_ago = float(sma200_arr[-21])

        # 52-week high/low
        tail = close.values[-_52W_DAYS:]
        high52w = float(tail.max())
        low52w = float(tail.min())

        rows.append({
            "code": str(code),
            "price": current_price,
            "sma50": smas.get("sma50", np.nan),
            "sma150": smas.get("sma150", np.nan),
            "sma200": smas.get("sma200", np.nan),
            "high52w": high52w,
            "low52w": low52w,
            "sma200_20d_ago": sma200_20d_ago,
        })

    return pd.DataFrame(rows)
