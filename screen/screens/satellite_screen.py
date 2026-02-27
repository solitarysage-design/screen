"""Satellite mode: strict Minervini TT + O'Neil EPS acceleration."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SATELLITE_YIELD_MIN = 0.02


def compute_eps_score(eps_q_list) -> tuple[float | None, bool | None, int]:
    """Compute O'Neil EPS score 0-10 from quarterly list (most-recent-first).

    Returns (eps_growth_yoy, eps_accel, eps_score).
    """
    _YOY_TIERS = [(0.50, 4), (0.25, 3), (0.10, 2), (0.00, 1)]
    _ACCEL_BONUS = 3

    if not isinstance(eps_q_list, list) or len(eps_q_list) < 5:
        return None, None, 0

    def _safe(v):
        try:
            f = float(v)
            return None if np.isnan(f) else f
        except Exception:
            return None

    q1 = _safe(eps_q_list[0])
    q5 = _safe(eps_q_list[4])

    yoy = None
    if q1 is not None and q5 is not None and q5 != 0:
        yoy = (q1 - q5) / abs(q5)

    # Acceleration
    accel = None
    if len(eps_q_list) >= 6:
        q2 = _safe(eps_q_list[1])
        q6 = _safe(eps_q_list[5])
        if all(v is not None for v in [q1, q2, q5, q6]) and q5 != 0 and q6 != 0:
            g_latest = (q1 - q5) / abs(q5)
            g_prior = (q2 - q6) / abs(q6)
            accel = g_latest > g_prior

    score = 0
    if yoy is not None and not (isinstance(yoy, float) and np.isinf(yoy)):
        for threshold, pts in _YOY_TIERS:
            if yoy >= threshold:
                score = pts
                break
    if accel is True:
        score = min(10, score + _ACCEL_BONUS)

    return yoy, accel, score


def apply_satellite_screen(
    df: pd.DataFrame,
    unknown_policy: str = "exclude",
) -> pd.DataFrame:
    """Apply Satellite (strict TT + O'Neil) screen.

    df must have: code, tt_all_pass (or tt_1..tt_8),
                  eps_q_list, dividend_yield_fwd
    Adds: satellite_pass, eps_growth_yoy, eps_accel, eps_score, satellite_drop_reasons
    """
    rows = []

    for _, row in df.iterrows():
        code = str(row.get("code", ""))
        drop_reasons: list[str] = []

        # Hard: Minervini TT all pass
        tt_pass = bool(row.get("tt_all_pass", False))
        if not tt_pass:
            drop_reasons.append("TT_not_all_pass")

        # Hard: minimum yield
        dy = row.get("dividend_yield_fwd")
        if dy is None or (isinstance(dy, float) and np.isnan(dy)):
            if unknown_policy == "exclude":
                drop_reasons.append("yield_unknown")
        elif float(dy) < SATELLITE_YIELD_MIN:
            drop_reasons.append("yield_below_2pct")

        # Scoring (no hard)
        eps_q = row.get("eps_q_list")
        yoy, accel, score = compute_eps_score(eps_q)

        satellite_pass = len(drop_reasons) == 0

        rows.append({
            "code": code,
            "satellite_pass": satellite_pass,
            "eps_growth_yoy": yoy,
            "eps_accel": accel,
            "eps_score": score,
            "satellite_drop_reasons": "; ".join(drop_reasons) if drop_reasons else None,
        })

    result = pd.DataFrame(rows)
    passed = result["satellite_pass"].sum()
    logger.info("Satellite screen: %d / %d pass", passed, len(df))

    if passed == 0:
        logger.warning("Satellite: 0\u4ef6\u901a\u904e \u2014 TT\u6761\u4ef6\u307e\u305f\u306f\u30c7\u30fc\u30bf\u3092\u78ba\u8a8d\u3057\u3066\u304f\u3060\u3055\u3044")

    return df.merge(result, on="code", how="left")
