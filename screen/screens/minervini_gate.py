"""Minervini Trend Template (TT) gate — 8 conditions."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from screen.config import (
    RS_MIN_PERCENTILE,
    TT_PRICE_VS_HIGH52W,
    TT_PRICE_VS_LOW52W,
)

logger = logging.getLogger(__name__)


def apply_minervini(tech: pd.DataFrame, rs: pd.DataFrame) -> pd.DataFrame:
    """Apply the 8-point Minervini Trend Template.

    Args:
        tech: Output of compute_technicals() — one row per code.
        rs: Output of compute_rs() — one row per code.

    Returns:
        DataFrame with columns:
            code, tt_1..tt_8, tt_all_pass
    """
    merged = tech.merge(rs[["code", "rs_percentile"]], on="code", how="left")

    def _flag(series: pd.Series) -> pd.Series:
        """Convert to boolean, treating NaN as False."""
        return series.fillna(False).astype(bool)

    # TT-1: price > SMA150 AND price > SMA200
    merged["tt_1"] = _flag(
        (merged["price"] > merged["sma150"]) & (merged["price"] > merged["sma200"])
    )

    # TT-2: SMA150 > SMA200
    merged["tt_2"] = _flag(merged["sma150"] > merged["sma200"])

    # TT-3: SMA200 is trending up (current > 20 trading days ago)
    merged["tt_3"] = _flag(merged["sma200"] > merged["sma200_20d_ago"])

    # TT-4: SMA50 > SMA150 AND SMA50 > SMA200
    merged["tt_4"] = _flag(
        (merged["sma50"] > merged["sma150"]) & (merged["sma50"] > merged["sma200"])
    )

    # TT-5: price > SMA50
    merged["tt_5"] = _flag(merged["price"] > merged["sma50"])

    # TT-6: price >= 52w low * 1.30
    merged["tt_6"] = _flag(merged["price"] >= merged["low52w"] * TT_PRICE_VS_LOW52W)

    # TT-7: price >= 52w high * 0.75
    merged["tt_7"] = _flag(merged["price"] >= merged["high52w"] * TT_PRICE_VS_HIGH52W)

    # TT-8: RS percentile >= threshold
    merged["tt_8"] = _flag(merged["rs_percentile"] >= RS_MIN_PERCENTILE)

    tt_cols = [f"tt_{i}" for i in range(1, 9)]
    merged["tt_all_pass"] = merged[tt_cols].all(axis=1)

    passed = merged["tt_all_pass"].sum()
    logger.info("Minervini gate: %d / %d pass", passed, len(merged))

    keep = ["code"] + tt_cols + ["tt_all_pass"]
    return merged[keep]
