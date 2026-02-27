"""Breakout and first-pullback detection."""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from screen.config import (
    BREAKOUT_LONG_DAYS,
    BREAKOUT_SHORT_DAYS,
    BREAKOUT_VOLUME_MULT,
    PULLBACK_REBOUND_DAYS,
    PULLBACK_SMA50_BAND,
    PULLBACK_WINDOW_DAYS,
)

logger = logging.getLogger(__name__)


def _detect_breakout(close: pd.Series, volume: pd.Series, window: int) -> bool:
    """Return True if the latest bar breaks out above the prior *window*-day high
    on elevated volume."""
    if len(close) < window + 1:
        return False

    prior_close = close.iloc[-(window + 1):-1]
    prior_vol = volume.iloc[-(window + 1):-1]

    current_close = close.iloc[-1]
    current_vol = volume.iloc[-1]

    if np.isnan(current_close) or np.isnan(current_vol):
        return False

    prior_max = prior_close.max()
    vol_avg = prior_vol.mean()

    return bool(current_close > prior_max and current_vol > vol_avg * BREAKOUT_VOLUME_MULT)


def _detect_breakout_within(
    close: pd.Series, volume: pd.Series, window: int, lookback: int
) -> int | None:
    """Return index offset (from end) of most recent breakout in last *lookback* bars,
    or None if no breakout found."""
    if len(close) < window + lookback:
        return None

    for offset in range(1, lookback + 1):
        idx = -offset
        sub_close = close.iloc[:len(close) - offset + 1]
        sub_volume = volume.iloc[:len(volume) - offset + 1]
        if _detect_breakout(sub_close, sub_volume, window):
            return offset

    return None


def compute_breakouts(prices: pd.DataFrame, asof: date) -> pd.DataFrame:
    """Compute breakout and first-pullback flags for each code as of *asof*.

    Returns DataFrame [code, breakout_20d, breakout_55d, first_pullback]
    """
    asof_ts = pd.Timestamp(asof)
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices[prices["Date"] <= asof_ts]

    rows = []
    for code, grp in prices.groupby("Code"):
        grp = grp.sort_values("Date").drop_duplicates(subset="Date", keep="last")
        close = grp["Close"].astype(float).reset_index(drop=True)
        volume = grp["Volume"].astype(float).reset_index(drop=True)

        # Current-day breakout flags
        bo_20 = _detect_breakout(close, volume, BREAKOUT_SHORT_DAYS)
        bo_55 = _detect_breakout(close, volume, BREAKOUT_LONG_DAYS)

        # First pullback: breakout occurred within PULLBACK_WINDOW_DAYS, now near SMA50
        first_pullback = False
        bo_offset = _detect_breakout_within(close, volume, BREAKOUT_SHORT_DAYS, PULLBACK_WINDOW_DAYS)

        if bo_offset is not None and bo_offset > 1:
            # Compute SMA50 at current time
            if len(close) >= 50:
                sma50 = close.tail(50).mean()
                current_price = close.iloc[-1]
                near_sma50 = abs(current_price - sma50) / sma50 <= PULLBACK_SMA50_BAND

                # Rebound: last N days trending up
                if len(close) >= PULLBACK_REBOUND_DAYS + 1:
                    rebound_slice = close.iloc[-PULLBACK_REBOUND_DAYS:]
                    rebound = rebound_slice.iloc[-1] > rebound_slice.iloc[0]
                else:
                    rebound = False

                first_pullback = bool(near_sma50 and rebound)

        rows.append({
            "code": str(code),
            "breakout_20d": bo_20,
            "breakout_55d": bo_55,
            "first_pullback": first_pullback,
        })

    return pd.DataFrame(rows)
