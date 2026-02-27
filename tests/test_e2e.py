"""End-to-end test using fixed sample data (no network calls)."""
from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# --- Sample data fixtures ---

def _sample_universe() -> pd.DataFrame:
    return pd.DataFrame({
        "Code": ["7203", "6758", "9984"],
        "Name": ["Toyota", "Sony", "SoftBank"],
        "MarketSegment": ["プライム", "プライム", "プライム"],
        "Sector17CodeName": ["輸送用機器", "電気機器", "情報通信・サービスその他"],
        "Sector33CodeName": ["輸送用機器", "電気機器", "情報・通信業"],
    })


def _sample_prices(n: int = 260) -> pd.DataFrame:
    import numpy as np
    frames = []
    for code, slope in [("7203", 2.0), ("6758", 3.0), ("9984", 1.0)]:
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        closes = [1000.0 + i * slope for i in range(n)]
        volumes = [1_000_000 + (i % 10) * 50_000 for i in range(n)]
        frames.append(pd.DataFrame({
            "Date": dates,
            "Code": code,
            "Close": closes,
            "Volume": volumes,
        }))
    return pd.concat(frames, ignore_index=True)


def _sample_fundamentals() -> pd.DataFrame:
    """New schema: cfo_annual/fcf_annual/div_paid_annual are total-yen lists."""
    return pd.DataFrame({
        "code": ["7203", "6758", "9984"],
        "eps_q_list": [
            [100, 90, 85, 80, 70, 65],  # accelerating
            [50, 45, 42, 38, 30, 28],
            [20, 18, 16, 14, 12, 10],
        ],
        # Total yen (realistic scale)
        "cfo_annual": [
            [3_000_000_000_000, 2_800_000_000_000, 2_600_000_000_000, 2_400_000_000_000, 2_200_000_000_000],
            [500_000_000_000, 480_000_000_000, 460_000_000_000, 440_000_000_000, 420_000_000_000],
            [100_000_000_000, 90_000_000_000, 80_000_000_000, 70_000_000_000, 60_000_000_000],
        ],
        "cfi_annual": [
            [-1_000_000_000_000, -900_000_000_000, -800_000_000_000, -700_000_000_000, -600_000_000_000],
            [-100_000_000_000, -90_000_000_000, -80_000_000_000, -70_000_000_000, -60_000_000_000],
            [-50_000_000_000, -45_000_000_000, -40_000_000_000, -35_000_000_000, -30_000_000_000],
        ],
        "fcf_annual": [
            [2_000_000_000_000, 1_900_000_000_000, 1_800_000_000_000, 1_700_000_000_000, 1_600_000_000_000],
            [400_000_000_000, 390_000_000_000, 380_000_000_000, 370_000_000_000, 360_000_000_000],
            [50_000_000_000, 45_000_000_000, 40_000_000_000, 35_000_000_000, 30_000_000_000],
        ],
        "div_paid_annual": [
            [800_000_000_000, 750_000_000_000, 700_000_000_000, 650_000_000_000, 600_000_000_000],
            [100_000_000_000, 90_000_000_000, 80_000_000_000, 70_000_000_000, 60_000_000_000],
            [20_000_000_000, 18_000_000_000, 16_000_000_000, 14_000_000_000, 12_000_000_000],
        ],
        "dps_actual_annual": [
            [120, 110, 100, 95, 90],   # growing → non_cut_years >= 4
            [60, 55, 50, 45, 40],
            [10, 9, 8, 7, 6],
        ],
        "dps_fwd": [130.0, 65.0, 11.0],
        "bps_latest": [5000.0, 3000.0, 1000.0],
        "eps_fwd": [800.0, 400.0, 100.0],
        "net_shares_latest": [1_450_000_000.0, 1_200_000_000.0, 2_000_000_000.0],
        "data_coverage": [1.0, 1.0, 1.0],
        "missing_fields": [[], [], []],
        "data_quality_flags": [[], [], []],
    })


# --- Tests ---

def test_compute_technicals_e2e():
    from screen.features.technicals import compute_technicals
    prices = _sample_prices()
    asof = prices["Date"].max().date()
    result = compute_technicals(prices, asof)
    assert len(result) == 3
    assert set(result["code"]) == {"7203", "6758", "9984"}
    assert result["price"].notna().all()


def test_compute_rs_e2e():
    from screen.features.relative_strength import compute_rs
    prices = _sample_prices()
    asof = prices["Date"].max().date()
    result = compute_rs(prices, topix=None, mode="universe_percentile", asof=asof)
    assert len(result) == 3
    assert result["rs_percentile"].between(0, 100).all()


def test_compute_fundamentals_metrics_e2e():
    from screen.features.fundamentals_metrics import compute_fundamentals_metrics
    fund = _sample_fundamentals()
    result = compute_fundamentals_metrics(fund)
    assert len(result) == 3
    # Toyota has growing dividends → non_cut_years >= 3
    toyota = result[result["code"] == "7203"].iloc[0]
    assert toyota["non_cut_years"] >= 3


def test_minervini_gate_e2e():
    from screen.features.technicals import compute_technicals
    from screen.features.relative_strength import compute_rs
    from screen.screens.minervini_gate import apply_minervini

    prices = _sample_prices(n=260)
    asof = prices["Date"].max().date()
    tech = compute_technicals(prices, asof)
    rs = compute_rs(prices, topix=None, mode="universe_percentile", asof=asof)
    result = apply_minervini(tech, rs)

    assert len(result) == 3
    for col in [f"tt_{i}" for i in range(1, 9)] + ["tt_all_pass"]:
        assert col in result.columns


def test_eps_score_e2e():
    from screen.screens.oniel_accel import compute_eps_score
    fund = _sample_fundamentals()
    result = compute_eps_score(fund)
    assert len(result) == 3
    # Accelerating EPS (Toyota) should score higher
    toyota = result[result["code"] == "7203"]["eps_score"].iloc[0]
    assert toyota >= 4  # YoY > 25% + accel bonus


def test_full_pipeline_produces_output():
    """Integration test: run full pipeline on sample data, assert output files created."""
    from screen.features.technicals import compute_technicals
    from screen.features.relative_strength import compute_rs
    from screen.features.breakouts import compute_breakouts
    from screen.features.fundamentals_metrics import compute_fundamentals_metrics
    from screen.screens.minervini_gate import apply_minervini
    from screen.screens.value_screen import compute_value_metrics
    from screen.screens.core_screen import apply_core_screen
    from screen.screens.satellite_screen import apply_satellite_screen

    prices = _sample_prices(n=260)
    asof = prices["Date"].max().date()
    fund = _sample_fundamentals()
    univ = _sample_universe()

    tech = compute_technicals(prices, asof)
    rs = compute_rs(prices, topix=None, mode="universe_percentile", asof=asof)
    bo = compute_breakouts(prices, asof)
    fm = compute_fundamentals_metrics(fund)
    tt = apply_minervini(tech, rs)

    univ_slim = univ[["Code", "Sector33CodeName", "Name"]].rename(columns={"Code": "code"})
    merged = (
        tech
        .merge(rs[["code", "rs_percentile", "ret_3m", "ret_6m", "ret_9m", "ret_12m"]], on="code", how="left")
        .merge(tt, on="code", how="left")
        .merge(fm.drop(columns=["eps_q_list"], errors="ignore"), on="code", how="left")
        .merge(univ_slim, on="code", how="left")
    )
    merged = compute_value_metrics(merged)
    merged = apply_core_screen(merged, unknown_policy="include")
    merged = merged.merge(fm[["code", "eps_q_list"]], on="code", how="left")
    merged = apply_satellite_screen(merged, unknown_policy="include")

    # Verify shapes and key columns
    assert len(merged) == 3
    assert "core_pass" in merged.columns
    assert "satellite_pass" in merged.columns

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "candidates.csv"
        merged.to_csv(out_path, index=False)
        assert out_path.exists()
        loaded = pd.read_csv(out_path)
        assert len(loaded) == 3
