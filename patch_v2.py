"""Standalone post-processing script to generate candidates_fixed_v2.csv.

Takes the existing candidates_fixed.csv (which has technical data for all stocks
but fundamentals only for TT-passing stocks), fetches fundamentals for the
missing holdings stocks, then applies all fixes:
  A) in_holdings = True for 21 holdings stocks
  B) fundamentals for the 17 holdings stocks that are TT-non-passing
  C) data_missing:<col> added to drop_reason for NaN core columns
  D) drop_reason NaN → "" (never NaN)
  E) writes candidates_fixed_v2.csv and holdings_debug_fixed_v2.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Setup path so we can import the screen package ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from screen.data.fundamentals import get_fundamentals
from screen.features.fundamentals_metrics import compute_fundamentals_metrics
from screen.screens.value_screen import compute_value_metrics
from screen.screens.core_screen import apply_core_screen
from screen.screens.satellite_screen import apply_satellite_screen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
import os
PROJECT_DIR    = Path(os.environ.get("SCREEN_PROJECT_DIR", "C:/Users/solit/projects/screen"))
OUTPUT_DIR     = Path(os.environ.get("SCREEN_OUTPUT_DIR", str(PROJECT_DIR / "output")))
HOLDINGS_CSV   = Path(os.environ.get("SCREEN_HOLDINGS_CSV", "C:/Users/solit/Downloads/holdings_extracted_20260224.csv"))
INPUT_CSV      = OUTPUT_DIR / "candidates_fixed.csv"
OUT_DIR        = OUTPUT_DIR
OUT_UNIVERSE   = OUT_DIR / "candidates_fixed_v2.csv"
OUT_HOLDINGS   = OUT_DIR / "holdings_debug_fixed_v2.csv"

# Core screening に必須の列 → NaN なら data_missing:<key> を drop_reason に追記
_CORE_REQUIRED_COLS: list[tuple[str, str]] = [
    ("non_cut_years_verified",   "non_cut_years"),
    ("dividend_yield_fwd_total", "yield_total"),
    ("cfo_pos_5y_ratio",         "cfo_pos_ratio"),
    ("fcf_pos_5y_ratio",         "fcf_pos_ratio"),
    ("fcf_payout_3y",            "fcf_payout"),
]

# Columns to carry over from newly fetched fm_df into the main df
_FM_COLS = [
    "code",
    "non_cut_years", "non_cut_years_verified", "non_cut_years_required",
    "coverage_years", "div_source_used", "needs_manual_dividend_check",
    "cfo_pos_5y_ratio", "fcf_pos_5y_ratio",
    "fcf_payout_3y", "fcf_payout_hard_fail", "fcf_latest", "div_paid_latest",
    "total_div_fwd", "total_div_actual",
    "dps_fwd", "bps_latest", "eps_fwd", "net_shares_latest",
    "metrics_coverage", "data_quality_flags",
]

# Columns for value_screen inputs that we need from the base df
_VALUE_INPUT_COLS = ["price", "dps_fwd", "bps_latest", "eps_fwd",
                     "total_div_fwd", "total_div_actual", "net_shares_latest",
                     "data_quality_flags", "fcf_latest", "Sector33CodeName", "Sector17CodeName"]


def _make_drop_reason(row: pd.Series) -> str:
    """Unified drop_reason: core + satellite + data_missing. Never returns NaN or empty string."""
    core_dr = row.get("core_drop_reasons")
    sat_dr  = row.get("satellite_drop_reasons")
    reasons: list[str] = []
    for dr in [core_dr, sat_dr]:
        # Skip None/NaN values; stringify only real strings
        if dr is None or (isinstance(dr, float) and np.isnan(dr)):
            continue
        for r in str(dr).split(";"):
            r = r.strip()
            if r and r.lower() != "nan" and r not in reasons:
                reasons.append(r)
    for col, key in _CORE_REQUIRED_COLS:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            tag = f"data_missing:{key}"
            if tag not in reasons:
                reasons.append(tag)
    # Return "OK" for truly passing stocks (empty string round-trips through CSV as NaN)
    return "; ".join(reasons) if reasons else "OK"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load holdings ────────────────────────────────────────────────────────
    h_df = pd.read_csv(HOLDINGS_CSV, dtype=str)
    holdings_codes: set[str] = set(h_df["code_jquants_5digit"].str.strip().dropna())
    logger.info("Holdings loaded: %d codes from %s", len(holdings_codes), HOLDINGS_CSV)
    assert len(holdings_codes) == 21, f"Expected 21 holdings, got {len(holdings_codes)}"

    # ── Load base CSV ────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    df["code"] = df["code"].astype(str).str.strip()
    logger.info("Base CSV loaded: %d rows, %d cols", len(df), len(df.columns))

    # ── A) Set in_holdings ───────────────────────────────────────────────────
    df["in_holdings"] = df["code"].isin(holdings_codes)
    n_ih = int(df["in_holdings"].sum())
    logger.info("in_holdings True: %d (expected 21)", n_ih)
    assert n_ih == 21, f"in_holdings assert failed: {n_ih}"

    # ── B) Identify which holdings stocks are missing fundamentals ───────────
    h_rows = df[df["in_holdings"]]
    missing_fund_mask = h_rows["non_cut_years_verified"].isna()
    missing_codes = h_rows.loc[missing_fund_mask, "code"].tolist()
    logger.info(
        "Holdings stocks missing fundamentals: %d / 21 → %s",
        len(missing_codes), missing_codes,
    )

    if missing_codes:
        # ── Fetch fundamentals for missing holdings ──────────────────────────
        logger.info("Fetching fundamentals for %d missing holdings stocks...", len(missing_codes))
        fund_raw = get_fundamentals(missing_codes)
        logger.info("Fundamentals fetched: %d rows", len(fund_raw))

        # ── Compute metrics ──────────────────────────────────────────────────
        fm_new = compute_fundamentals_metrics(fund_raw)
        logger.info("Metrics computed: %d rows", len(fm_new))

        # ── Update rows in df for the missing stocks ─────────────────────────
        fm_cols_present = [c for c in _FM_COLS if c in fm_new.columns]
        fm_new = fm_new[fm_cols_present].copy()
        fm_new["code"] = fm_new["code"].astype(str).str.strip()

        # Drop the old (all-NaN) fm columns for these codes and re-merge
        fm_base_cols = [c for c in fm_cols_present if c != "code"]
        # Zero out existing values for missing_codes rows
        for col in fm_base_cols:
            if col in df.columns:
                df.loc[df["code"].isin(missing_codes), col] = np.nan

        # Update row-by-row from fm_new
        # Use .at[] for scalar, and object assignment for list columns to avoid broadcast error
        fm_new_indexed = fm_new.set_index("code")
        for code in missing_codes:
            if code not in fm_new_indexed.index:
                logger.warning("No fundamentals returned for %s", code)
                continue
            fm_row = fm_new_indexed.loc[code]
            idx_positions = df.index[df["code"] == code].tolist()
            if not idx_positions:
                continue
            row_idx = idx_positions[0]
            for col in fm_base_cols:
                if col in df.columns and col in fm_row.index:
                    val = fm_row[col]
                    # list/dict columns must be assigned to object dtype cell directly
                    if isinstance(val, (list, dict)):
                        df.at[row_idx, col] = val
                    else:
                        df.at[row_idx, col] = val

        logger.info("Fundamentals merged into main df for %d stocks", len(missing_codes))

        # ── Recompute value metrics for updated rows ─────────────────────────
        # We need price + new fundamentals columns to recompute dividend_yield_fwd_total etc.
        # Build a sub-df with the updated rows and run compute_value_metrics
        recompute_mask = df["code"].isin(missing_codes)
        val_input_cols = ["code"] + [c for c in _VALUE_INPUT_COLS if c in df.columns]
        sub_val = df.loc[recompute_mask, val_input_cols].copy()

        if not sub_val.empty:
            sub_val_out = compute_value_metrics(sub_val)
            # Columns added by value_screen
            val_new_cols = [
                "dividend_yield_fwd", "dividend_yield_fwd_total",
                "yield_split_mismatch", "high_yield_risk",
                "per_fwd", "pbr", "fcf_yield", "per_sector_rank", "value_pass",
                "data_quality_flags",
            ]
            for col in val_new_cols:
                if col not in df.columns:
                    df[col] = np.nan
            sub_val_out_indexed = sub_val_out.set_index("code")
            for code in missing_codes:
                if code not in sub_val_out_indexed.index:
                    continue
                row_val = sub_val_out_indexed.loc[code]
                idx_pos = df.index[df["code"] == code].tolist()
                if not idx_pos:
                    continue
                row_idx = idx_pos[0]
                for col in val_new_cols:
                    if col in sub_val_out.columns:
                        df.at[row_idx, col] = row_val[col]
            logger.info("Value metrics recomputed for %d stocks", len(missing_codes))

        # ── Recompute core_screen for updated rows ───────────────────────────
        # Build sub-df with all core input columns for the missing stocks
        core_input_cols = [
            "code", "non_cut_years_verified", "non_cut_years", "coverage_years",
            "cfo_pos_5y_ratio", "fcf_pos_5y_ratio", "fcf_payout_3y",
            "fcf_payout_hard_fail", "dividend_yield_fwd_total", "dividend_yield_fwd",
            "value_pass", "price", "sma50", "sma200", "sma200_20d_ago",
            "low52w", "high52w", "rs_percentile", "tt_all_pass",
            "Sector33CodeName", "Sector17CodeName",
        ]
        core_cols_present = [c for c in core_input_cols if c in df.columns]
        sub_core = df.loc[recompute_mask, core_cols_present].copy()

        if not sub_core.empty:
            sub_core_out = apply_core_screen(sub_core)
            core_new_cols = [
                "core_pass", "core_candidate", "core_momo_pass",
                "trend_score", "core_drop_reasons",
            ]
            for col in core_new_cols:
                if col not in df.columns:
                    df[col] = np.nan
            if "core_pass" in sub_core_out.columns:
                sub_core_indexed = sub_core_out.set_index("code")
                for code in missing_codes:
                    if code not in sub_core_indexed.index:
                        continue
                    row_core = sub_core_indexed.loc[code]
                    idx_pos = df.index[df["code"] == code].tolist()
                    if not idx_pos:
                        continue
                    row_idx = idx_pos[0]
                    for col in core_new_cols:
                        if col in sub_core_out.columns:
                            df.at[row_idx, col] = row_core[col]
            logger.info("Core screen recomputed for %d stocks", len(missing_codes))

        # ── Recompute satellite for updated rows ─────────────────────────────
        sat_input_cols = ["code", "tt_all_pass", "dividend_yield_fwd"]
        # eps_q_list column - may be present or absent
        if "eps_q_list" in df.columns:
            sat_input_cols.append("eps_q_list")
        sat_cols_present = [c for c in sat_input_cols if c in df.columns]
        sub_sat = df.loc[recompute_mask, sat_cols_present].copy()

        if not sub_sat.empty:
            sub_sat_out = apply_satellite_screen(sub_sat)
            sat_new_cols = [
                "satellite_pass", "eps_growth_yoy", "eps_accel", "eps_score",
                "satellite_drop_reasons",
            ]
            for col in sat_new_cols:
                if col not in df.columns:
                    df[col] = np.nan
            if "satellite_pass" in sub_sat_out.columns:
                sub_sat_indexed = sub_sat_out.set_index("code")
                for code in missing_codes:
                    if code not in sub_sat_indexed.index:
                        continue
                    row_sat = sub_sat_indexed.loc[code]
                    idx_pos = df.index[df["code"] == code].tolist()
                    if not idx_pos:
                        continue
                    row_idx = idx_pos[0]
                    for col in sat_new_cols:
                        if col in sub_sat_out.columns:
                            df.at[row_idx, col] = row_sat[col]
            logger.info("Satellite screen recomputed for %d stocks", len(missing_codes))

    # ── C+D) Recompute drop_reason for all rows ──────────────────────────────
    df["drop_reason"] = df.apply(_make_drop_reason, axis=1)
    n_dr_nan = int(df["drop_reason"].isna().sum())
    logger.info("drop_reason NaN after fix: %d (should be 0)", n_dr_nan)
    assert n_dr_nan == 0, f"drop_reason still has {n_dr_nan} NaN rows"

    # ── Validation F ─────────────────────────────────────────────────────────
    logger.info("=== Validation ===")
    logger.info("in_holdings True: %d (expected 21)", int(df["in_holdings"].sum()))

    h_mask = df["in_holdings"]
    for col, key in _CORE_REQUIRED_COLS:
        if col in df.columns:
            n_nan = int(df.loc[h_mask, col].isna().sum())
            status = "OK" if n_nan == 0 else "WARNING"
            logger.info("[%s] 保有株 %s NaN: %d / 21", status, col, n_nan)

    logger.info("core_pass: %d件 / %d", int(df["core_pass"].sum()), len(df))

    dm_counts = (
        df["drop_reason"]
        .fillna("")
        .str.split("; ")
        .explode()
        .str.strip()
        .pipe(lambda s: s[s.str.startswith("data_missing:")])
        .value_counts()
        .head(10)
    )
    if not dm_counts.empty:
        logger.info("data_missing 上位:\n%s", dm_counts.to_string())
    else:
        logger.info("data_missing: 0件")

    # ── E) Save outputs ──────────────────────────────────────────────────────
    # Stringify list columns for CSV
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], list):
                df[col] = df[col].apply(
                    lambda v: "; ".join(str(x) for x in v) if isinstance(v, list) else v
                )

    df.to_csv(OUT_UNIVERSE, index=False, encoding="utf-8-sig")
    logger.info("candidates_fixed_v2.csv → %s (%d rows)", OUT_UNIVERSE, len(df))

    # holdings_debug_fixed_v2.csv: 保有株のみ、必須列
    debug_cols = [
        "code", "in_holdings", "core_pass", "core_candidate",
        "satellite_pass", "drop_reason", "core_drop_reasons",
        "dividend_yield_fwd_total", "dividend_yield_fwd",
        "non_cut_years_verified", "coverage_years",
        "cfo_pos_5y_ratio", "fcf_pos_5y_ratio", "fcf_payout_3y",
        "value_pass", "data_quality_flags",
        "rs_percentile", "tt_all_pass",
    ]
    debug_cols_present = [c for c in debug_cols if c in df.columns]
    h_debug = df[df["in_holdings"]][debug_cols_present].copy()

    # Merge in name + account from holdings CSV
    h_df["code_jquants_5digit"] = h_df["code_jquants_5digit"].str.strip()
    h_debug = h_df[["code_jquants_5digit", "name_pdf", "account", "shares", "avg_cost_yen"]].merge(
        h_debug,
        left_on="code_jquants_5digit",
        right_on="code",
        how="left",
    )

    h_debug.to_csv(OUT_HOLDINGS, index=False, encoding="utf-8-sig")
    logger.info("holdings_debug_fixed_v2.csv → %s (%d rows)", OUT_HOLDINGS, len(h_debug))

    # Final assert
    assert int(df["in_holdings"].sum()) == 21, "in_holdings final assert failed"
    assert int(df["drop_reason"].isna().sum()) == 0, "drop_reason NaN final assert failed"
    logger.info("=== All assertions passed ===")


if __name__ == "__main__":
    main()
