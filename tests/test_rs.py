"""Unit tests for relative_strength.py."""
import numpy as np
import pandas as pd
import pytest
from datetime import date

from screen.features.relative_strength import compute_rs


def _make_prices(code: str, n: int = 300, slope: float = 1.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = [1000.0 + i * slope for i in range(n)]
    return pd.DataFrame({
        "Date": dates,
        "Code": code,
        "Close": prices,
        "Volume": [100_000] * n,
    })


def test_percentile_range_0_100():
    """rs_percentile should be in [0, 100] for all codes."""
    df = pd.concat([
        _make_prices(f"{i:04d}", slope=float(i)) for i in range(1, 11)
    ], ignore_index=True)
    asof = df["Date"].max().date()
    result = compute_rs(df, topix=None, mode="universe_percentile", asof=asof)

    assert not result.empty
    assert result["rs_percentile"].min() >= 0.0
    assert result["rs_percentile"].max() <= 100.0


def test_stronger_stock_higher_percentile():
    """A stock with higher returns should have higher rs_percentile."""
    df = pd.concat([
        _make_prices("weak", slope=0.1),
        _make_prices("strong", slope=5.0),
    ], ignore_index=True)
    asof = df["Date"].max().date()
    result = compute_rs(df, topix=None, mode="universe_percentile", asof=asof)

    weak = result[result["code"] == "weak"]["rs_percentile"].iloc[0]
    strong = result[result["code"] == "strong"]["rs_percentile"].iloc[0]
    assert strong > weak


def test_topix_mode_uses_relative_return():
    """In topix mode, rs_score should reflect relative performance."""
    stock = _make_prices("stock", slope=3.0)
    topix = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=300, freq="B"),
        "Close": [1000.0 + i * 1.0 for i in range(300)],
    })
    asof = stock["Date"].max().date()
    result = compute_rs(stock, topix=topix, mode="topix", asof=asof)

    assert not result.empty
    # Outperforming stock should have positive rs_score
    row = result[result["code"] == "stock"].iloc[0]
    assert row["rs_score"] > 0


def test_insufficient_data_returns_nan():
    """Codes with too few bars should return NaN for rs_score."""
    df = _make_prices("short", n=10)
    asof = df["Date"].max().date()
    result = compute_rs(df, topix=None, mode="universe_percentile", asof=asof)
    row = result[result["code"] == "short"].iloc[0]
    assert np.isnan(row["rs_score"])


def test_output_columns():
    """Result must contain required columns."""
    df = _make_prices("1234")
    asof = df["Date"].max().date()
    result = compute_rs(df, topix=None, mode="universe_percentile", asof=asof)
    for col in ["code", "rs_score", "rs_percentile"]:
        assert col in result.columns
