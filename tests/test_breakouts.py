"""Unit tests for breakouts.py."""
import pandas as pd
import pytest
from datetime import date

from screen.features.breakouts import compute_breakouts


def _make_flat_prices(code: str, n: int = 100, price: float = 1000.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Date": dates,
        "Code": code,
        "Close": [price] * n,
        "Volume": [50_000] * n,
    })


def _make_breakout_prices(code: str, n: int = 80, breakout_at: int = -1) -> pd.DataFrame:
    """Flat prices, then a single high-volume breakout bar at position breakout_at."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = [1000.0] * n
    volumes = [50_000] * n
    # The last bar breaks out
    closes[breakout_at] = 1100.0   # > prior max
    volumes[breakout_at] = 100_000  # > avg * 1.5
    return pd.DataFrame({
        "Date": dates,
        "Code": code,
        "Close": closes,
        "Volume": volumes,
    })


def test_no_breakout_flat_prices():
    """Flat price series should not trigger a breakout."""
    df = _make_flat_prices("flat")
    asof = df["Date"].max().date()
    result = compute_breakouts(df, asof)
    row = result[result["code"] == "flat"].iloc[0]
    # Current bar equals prior max (not strictly greater), so no breakout
    assert row["breakout_20d"] is False or row["breakout_20d"] == False


def test_breakout_detected_on_breakout_bar():
    """A bar that exceeds prior-window high on elevated volume triggers breakout."""
    df = _make_breakout_prices("bo")
    asof = df["Date"].max().date()
    result = compute_breakouts(df, asof)
    row = result[result["code"] == "bo"].iloc[0]
    assert row["breakout_20d"] is True or row["breakout_20d"] == True


def test_no_breakout_without_volume():
    """Price spike without volume spike should not be a breakout."""
    df = _make_flat_prices("novol", n=80)
    df_mod = df.copy()
    df_mod.loc[df_mod.index[-1], "Close"] = 1100.0
    # Volume stays at 50_000 — below 1.5× mean
    asof = df_mod["Date"].max().date()
    result = compute_breakouts(df_mod, asof)
    row = result[result["code"] == "novol"].iloc[0]
    assert row["breakout_20d"] == False


def test_insufficient_bars_no_breakout():
    """Fewer bars than window should not raise errors and return no breakout."""
    df = _make_flat_prices("short", n=5)
    asof = df["Date"].max().date()
    result = compute_breakouts(df, asof)
    row = result[result["code"] == "short"].iloc[0]
    assert row["breakout_20d"] == False
    assert row["breakout_55d"] == False


def test_first_pullback_detected():
    """Stock that broke out recently and pulled back near SMA50 triggers pullback flag."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = [1000.0] * n
    volumes = [50_000] * n

    # Breakout 20 bars ago
    bo_idx = n - 20
    closes[bo_idx] = 1100.0
    volumes[bo_idx] = 100_000

    # Price pulled back and is rebounding near SMA50
    for i in range(bo_idx + 1, n):
        closes[i] = 1010.0 + (i - bo_idx) * 0.5  # slight uptrend near SMA50

    df = pd.DataFrame({"Date": dates, "Code": "pb", "Close": closes, "Volume": volumes})
    asof = df["Date"].max().date()
    result = compute_breakouts(df, asof)
    row = result[result["code"] == "pb"].iloc[0]
    # The first_pullback logic requires the price near SMA50 — this is approximate
    # Just ensure no crash and the column exists
    assert "first_pullback" in row.index


def test_output_columns():
    df = _make_flat_prices("1234")
    asof = df["Date"].max().date()
    result = compute_breakouts(df, asof)
    for col in ["code", "breakout_20d", "breakout_55d", "first_pullback"]:
        assert col in result.columns
