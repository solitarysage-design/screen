"""Unit tests for technicals.py."""
import numpy as np
import pandas as pd
import pytest
from datetime import date

from screen.features.technicals import compute_technicals


def _make_prices(code: str, n: int = 300, start_price: float = 1000.0) -> pd.DataFrame:
    """Generate deterministic uptrending price series."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = [start_price + i * 1.5 for i in range(n)]
    return pd.DataFrame({
        "Date": dates,
        "Code": code,
        "Close": prices,
        "Volume": [100_000] * n,
    })


def test_sma_values():
    """SMA50/150/200 should equal simple rolling mean."""
    df = _make_prices("1001", n=250)
    asof = df["Date"].max().date()
    result = compute_technicals(df, asof)

    row = result[result["code"] == "1001"].iloc[0]
    close = df["Close"].values

    np.testing.assert_allclose(row["sma50"], np.mean(close[-50:]), rtol=1e-6)
    np.testing.assert_allclose(row["sma150"], np.mean(close[-150:]), rtol=1e-6)


def test_sma200_nan_when_insufficient_data():
    """SMA200 should be NaN when fewer than 200 bars available."""
    df = _make_prices("1002", n=100)
    asof = df["Date"].max().date()
    result = compute_technicals(df, asof)
    row = result[result["code"] == "1002"].iloc[0]
    assert np.isnan(row["sma200"])


def test_52w_high_low():
    """52-week high/low should reflect last 252 bars."""
    df = _make_prices("1003", n=300, start_price=500.0)
    asof = df["Date"].max().date()
    result = compute_technicals(df, asof)
    row = result[result["code"] == "1003"].iloc[0]

    tail_252 = df["Close"].tail(252)
    assert row["high52w"] == pytest.approx(tail_252.max())
    assert row["low52w"] == pytest.approx(tail_252.min())


def test_asof_filters_future_data():
    """Prices after asof should be excluded."""
    df = _make_prices("1004", n=260)
    # Use a date 50 bars before the end
    asof = df["Date"].iloc[200].date()
    result = compute_technicals(df, asof)
    row = result[result["code"] == "1004"].iloc[0]

    expected_price = df[df["Date"] <= pd.Timestamp(asof)]["Close"].iloc[-1]
    assert row["price"] == pytest.approx(expected_price)


def test_multiple_codes():
    """compute_technicals should return one row per code."""
    df = pd.concat([
        _make_prices("1111", n=250),
        _make_prices("2222", n=250),
        _make_prices("3333", n=250),
    ], ignore_index=True)
    asof = df["Date"].max().date()
    result = compute_technicals(df, asof)
    assert set(result["code"].tolist()) == {"1111", "2222", "3333"}
    assert len(result) == 3
