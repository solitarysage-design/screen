"""O'Neil EPS acceleration score (0-10)."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Score thresholds
_YOY_TIERS = [
    (0.50, 4),   # YoY ≥ 50% → 4 points
    (0.25, 3),   # YoY ≥ 25% → 3 points
    (0.10, 2),   # YoY ≥ 10% → 2 points
    (0.00, 1),   # YoY ≥  0% → 1 point
]
_ACCEL_BONUS = 3  # bonus if acceleration vs prior quarter


def _latest_yoy_growth(eps_q: list[float | None]) -> float | None:
    """Compute YoY EPS growth from quarterly list (most-recent-first).

    Compares Q1 (most recent) to Q5 (same quarter last year).
    """
    if eps_q is None or len(eps_q) < 5:
        return None
    q1 = eps_q[0]
    q5 = eps_q[4]
    if q1 is None or q5 is None:
        return None
    if q5 == 0:
        return None if q1 == 0 else float("inf") if q1 > 0 else float("-inf")
    return (q1 - q5) / abs(q5)


def _is_accelerating(eps_q: list[float | None]) -> bool | None:
    """Compare latest QoQ growth rate to prior QoQ.

    Q1 vs Q5 (YoY) compared to Q2 vs Q6 (prior YoY).
    Returns None if insufficient data.
    """
    if eps_q is None or len(eps_q) < 6:
        return None

    q1, q2, q5, q6 = eps_q[0], eps_q[1], eps_q[4], eps_q[5]
    if any(v is None for v in [q1, q2, q5, q6]):
        return None
    if q5 == 0 or q6 == 0:
        return None

    growth_latest = (q1 - q5) / abs(q5)
    growth_prior = (q2 - q6) / abs(q6)

    return growth_latest > growth_prior


def compute_eps_score(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Compute O'Neil EPS acceleration score (0-10) per code.

    Args:
        fundamentals: Output of get_fundamentals() with eps_q_list column.

    Returns:
        DataFrame [code, eps_growth_yoy, eps_accel, eps_score]
    """
    rows = []

    for _, row in fundamentals.iterrows():
        code = str(row.get("code", ""))
        eps_q: list[float | None] | None = row.get("eps_q_list")

        yoy = _latest_yoy_growth(eps_q)
        accel = _is_accelerating(eps_q)

        # Base score from YoY growth
        score = 0
        if yoy is not None and not (isinstance(yoy, float) and np.isinf(yoy)):
            for threshold, points in _YOY_TIERS:
                if yoy >= threshold:
                    score = points
                    break

        # Acceleration bonus
        if accel is True:
            score = min(10, score + _ACCEL_BONUS)

        rows.append({
            "code": code,
            "eps_growth_yoy": yoy,
            "eps_accel": accel,
            "eps_score": score,
        })

    return pd.DataFrame(rows)
