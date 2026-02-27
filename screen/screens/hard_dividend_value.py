"""Hard dividend-value filters (増配バリュー母集団)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from screen.config import (
    CYCLICAL_SECTOR33,
    FCF_PAYOUT_CYCLICAL,
    FCF_PAYOUT_HARD,
    UNKNOWN_POLICY,
)

logger = logging.getLogger(__name__)

# Minimum thresholds
MIN_NON_CUT_YEARS = 3           # Condition A: at least N non-cut years
MIN_CFO_POS_RATIO = 0.60        # Condition B: ≥60% of last 5y CFO positive
MIN_FCF_POS_RATIO = 0.60        # Condition C: ≥60% of last 5y FCF positive
MAX_FCF_PAYOUT_NON_CYC = FCF_PAYOUT_HARD      # Condition D (non-cyclical)
MAX_FCF_PAYOUT_CYC = FCF_PAYOUT_CYCLICAL      # Condition D (cyclical)
# Condition E: net_debt/EBITDA trend — optional / not enforced by default


def _is_cyclical(sector33: str | None) -> bool:
    if sector33 is None:
        return False
    return sector33 in CYCLICAL_SECTOR33


def _check(
    value: float | None,
    threshold: float,
    operator: str,
    policy: str,
) -> tuple[bool, bool]:
    """Evaluate a single condition.

    Returns:
        (pass, is_unknown)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        is_unknown = True
        passes = policy == "include"
        return passes, is_unknown

    if operator == ">=":
        return value >= threshold, False
    elif operator == "<=":
        return value <= threshold, False
    elif operator == ">":
        return value > threshold, False
    return False, False


def apply_hard_filters(
    metrics: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    min_yield: float = 0.03,
    policy: str = UNKNOWN_POLICY,
) -> pd.DataFrame:
    """Apply hard dividend-value filters.

    Args:
        metrics: Output of compute_fundamentals_metrics() merged with tech/price data.
                 Must include: code, non_cut_years, cfo_pos_5y_ratio, fcf_pos_5y_ratio,
                 fcf_payout_3y_avg. Optionally: div_yield, sector33.
        universe: Universe DataFrame with Sector33CodeName for cyclical classification.
        min_yield: Minimum dividend yield threshold.
        policy: "exclude" | "include" — how to handle unknown (NaN) values.

    Returns:
        DataFrame with added columns: hd_pass, drop_reason, hd_unknown_flags
    """
    df = metrics.copy()

    # Merge sector info if available
    if universe is not None and "Sector33CodeName" in universe.columns:
        code_col = "code" if "code" in universe.columns else "Code"
        sector_map = universe.set_index(code_col)["Sector33CodeName"].to_dict()
        if "code" in df.columns:
            df["sector33"] = df["code"].map(sector_map)
    elif "sector33" not in df.columns:
        df["sector33"] = None

    results = []
    for _, row in df.iterrows():
        code = row.get("code", "")
        sector33 = row.get("sector33")
        cyclical = _is_cyclical(sector33)
        unknown_flags: list[str] = []
        drop_reason: str | None = None
        passed = True

        # --- Condition A: Non-cut years ---
        passes_a, unk_a = _check(row.get("non_cut_years"), MIN_NON_CUT_YEARS, ">=", policy)
        if unk_a:
            unknown_flags.append("A_non_cut_years")
        if not passes_a:
            passed = False
            drop_reason = drop_reason or "A_non_cut_years"

        # --- Condition B: CFO positive ratio ---
        passes_b, unk_b = _check(row.get("cfo_pos_5y_ratio"), MIN_CFO_POS_RATIO, ">=", policy)
        if unk_b:
            unknown_flags.append("B_cfo_pos_ratio")
        if not passes_b and drop_reason is None:
            passed = False
            drop_reason = "B_cfo_pos_ratio"

        # --- Condition C: FCF positive ratio ---
        passes_c, unk_c = _check(row.get("fcf_pos_5y_ratio"), MIN_FCF_POS_RATIO, ">=", policy)
        if unk_c:
            unknown_flags.append("C_fcf_pos_ratio")
        if not passes_c and drop_reason is None:
            passed = False
            drop_reason = "C_fcf_pos_ratio"

        # --- Condition D: FCF payout ratio ---
        payout_cap = MAX_FCF_PAYOUT_CYC if cyclical else MAX_FCF_PAYOUT_NON_CYC
        payout = row.get("fcf_payout_3y_avg")
        passes_d, unk_d = _check(payout, payout_cap, "<=", policy)
        if unk_d:
            unknown_flags.append("D_fcf_payout")
        if not passes_d and drop_reason is None:
            passed = False
            drop_reason = "D_fcf_payout"

        # --- Condition E: dividend yield (optional, if column present) ---
        if "div_yield" in row.index:
            passes_e, unk_e = _check(row.get("div_yield"), min_yield, ">=", policy)
            if unk_e:
                unknown_flags.append("E_div_yield")
            if not passes_e and drop_reason is None:
                passed = False
                drop_reason = "E_div_yield"

        results.append({
            "code": code,
            "hd_pass": passed,
            "drop_reason": drop_reason if not passed else None,
            "hd_unknown_flags": ",".join(unknown_flags) if unknown_flags else None,
        })

    result_df = pd.DataFrame(results)
    merged = df.merge(result_df, on="code", how="left")

    passed_count = merged["hd_pass"].sum()
    logger.info("Hard dividend filter: %d / %d pass", passed_count, len(merged))

    return merged
