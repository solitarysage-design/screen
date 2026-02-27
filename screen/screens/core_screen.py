"""Core mode: dividend safety (strict) — trend conditions は scoring のみ、hard 条件ではない。

core_pass hard 条件:
  A) non_cut_years_verified >= 5 かつ coverage_years >= 5
  B) cfo_pos_5y_ratio >= 0.80
  C) fcf_pos_5y_ratio >= 0.60
  D) fcf_payout_3y <= 0.70 (景気敏感 <= 0.60)
  E) dividend_yield_fwd_total (なければ dividend_yield_fwd) >= 0.03
  F) value_pass = True

coverage_years < 5 の場合:
  - core_pass = False（配当履歴が不十分で検証不可）
  - core_candidate = True（他条件クリアなら候補として可視化）
  - needs_manual_dividend_check = True（手動確認推奨）

トレンド条件 (T1〜T6) は drop_reasons に記録するが core_pass には影響しない。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from screen.config import CYCLICAL_SECTOR33, FINANCIAL_SECTOR33
from screen.features.fundamentals_metrics import NON_CUT_YEARS_REQUIRED

logger = logging.getLogger(__name__)

# Core hard thresholds
CORE_NON_CUT_YEARS_MIN = NON_CUT_YEARS_REQUIRED  # = 5
CORE_CFO_POS_RATIO_MIN = 0.80
CORE_FCF_POS_RATIO_MIN = 0.60
CORE_FCF_PAYOUT_NON_CYC = 0.70
CORE_FCF_PAYOUT_CYC = 0.60
CORE_YIELD_MIN = 0.03

# Trend (informational only — not hard for Core)
CORE_PRICE_VS_LOW52W = 1.20   # +20% above 52w low
CORE_PRICE_VS_HIGH52W = 0.60  # within 40% of 52w high
CORE_RS_MIN = 70.0


def _is_cyclical(sector) -> bool:
    if sector is None:
        return False
    return str(sector) in CYCLICAL_SECTOR33


def _is_financial(sector) -> bool:
    """True for sectors where CFO/FCF metrics are structurally inapplicable."""
    if sector is None:
        return False
    return str(sector) in FINANCIAL_SECTOR33


def _check(val, threshold, op, policy="exclude") -> tuple[bool, bool]:
    """Return (passes, is_unknown). op: '>=' or '<='."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return (policy == "include"), True
    if op == ">=":
        return float(val) >= threshold, False
    if op == "<=":
        return float(val) <= threshold, False
    return False, False


def apply_core_screen(
    df: pd.DataFrame,
    unknown_policy: str = "exclude",
) -> pd.DataFrame:
    """Apply Core mode hard conditions (fundamentals only).

    df must have: code, non_cut_years_verified (or non_cut_years), coverage_years,
                  cfo_pos_5y_ratio, fcf_pos_5y_ratio,
                  fcf_payout_3y, fcf_payout_hard_fail,
                  dividend_yield_fwd_total (preferred) or dividend_yield_fwd,
                  value_pass,
                  price, sma50, sma200, sma200_20d_ago, low52w, high52w,
                  rs_percentile,
                  sector33 or Sector33CodeName (optional),
                  needs_manual_dividend_check (optional)

    Adds: is_financial, core_pass, core_fin_pass, core_candidate, core_drop_reasons

    core_pass = True  → non-financial, all fundamental conditions met, coverage_years >= 5
    core_fin_pass = True → financial sector, alternative conditions met (no CFO/FCF)
    core_candidate = True → coverage_years < 5 but all OTHER conditions pass
    core_drop_reasons contains all fail reasons for transparency.
    """
    result_rows = []

    sector_col = next(
        (c for c in ["Sector33CodeName", "sector33"] if c in df.columns), None
    )

    for _, row in df.iterrows():
        code = str(row.get("code", ""))
        sector = row.get(sector_col) if sector_col else None
        cyclical = _is_cyclical(sector)
        financial = _is_financial(sector)
        payout_cap = CORE_FCF_PAYOUT_CYC if cyclical else CORE_FCF_PAYOUT_NON_CYC

        fund_drops: list[str] = []  # fundamental failures → affect core_pass

        # ── Coverage check ────────────────────────────────────────────────
        non_cut_val = row.get("non_cut_years_verified")
        if non_cut_val is None or (isinstance(non_cut_val, float) and np.isnan(non_cut_val)):
            non_cut_val = row.get("non_cut_years")

        coverage_years = row.get("coverage_years")
        if coverage_years is None or (isinstance(coverage_years, float) and np.isnan(coverage_years)):
            coverage_years = 0
        coverage_years = int(coverage_years)

        coverage_insufficient = coverage_years < CORE_NON_CUT_YEARS_MIN

        # --- Condition A: Non-cut years ---
        passes_a, _ = _check(non_cut_val, CORE_NON_CUT_YEARS_MIN, ">=", unknown_policy)
        if not passes_a:
            fund_drops.append("A_non_cut_years")
        if coverage_insufficient:
            if "A_non_cut_years" not in fund_drops:
                fund_drops.append("A_coverage_insufficient")

        # --- Conditions B/C/D: CFO/FCF (skipped for financial sector) ---
        if not financial:
            passes_b, _ = _check(
                row.get("cfo_pos_5y_ratio"), CORE_CFO_POS_RATIO_MIN, ">=", unknown_policy
            )
            if not passes_b:
                fund_drops.append("B_cfo_pos_ratio")

            passes_c, _ = _check(
                row.get("fcf_pos_5y_ratio"), CORE_FCF_POS_RATIO_MIN, ">=", unknown_policy
            )
            if not passes_c:
                fund_drops.append("C_fcf_pos_ratio")

            passes_d, _ = _check(row.get("fcf_payout_3y"), payout_cap, "<=", unknown_policy)
            if not passes_d:
                fund_drops.append("D_fcf_payout")

            if row.get("fcf_payout_hard_fail") is True:
                fund_drops.append("D_fcf_hard_fail")
        else:
            fund_drops.append("fin_sector_excluded_from_core")

        # --- Condition E: dividend yield ---
        yield_val = row.get("dividend_yield_fwd_total")
        if yield_val is None or (isinstance(yield_val, float) and np.isnan(yield_val)):
            yield_val = row.get("dividend_yield_fwd")
        passes_e, _ = _check(yield_val, CORE_YIELD_MIN, ">=", unknown_policy)
        if not passes_e:
            fund_drops.append("E_div_yield")

        # --- Condition F: value_pass ---
        if not row.get("value_pass", False):
            fund_drops.append("F_value_pass")

        # ── core_pass / core_candidate ────────────────────────────────────
        # Financial stocks always get core_pass=False (handled via core_fin_pass)
        core_pass = (not financial) and len(fund_drops) == 0

        non_coverage_drops = [
            d for d in fund_drops
            if d not in ("A_non_cut_years", "A_coverage_insufficient")
        ]
        core_candidate = (
            not core_pass
            and not financial
            and coverage_insufficient
            and len([d for d in non_coverage_drops if d != "fin_sector_excluded_from_core"]) == 0
        )

        # ── core_fin_pass: financial sector alternative conditions ─────────
        # Conditions: A(non_cut>=5) + E(yield>=3%) + F(value_pass) + eps_fwd>0
        fin_drops: list[str] = []
        if financial:
            if not passes_a:
                fin_drops.append("A_non_cut_years")
            if coverage_insufficient and "A_non_cut_years" not in fin_drops:
                fin_drops.append("A_coverage_insufficient")
            if not passes_e:
                fin_drops.append("E_div_yield")
            if not row.get("value_pass", False):
                fin_drops.append("F_value_pass")
            eps_fwd = row.get("eps_fwd")
            if eps_fwd is None or (isinstance(eps_fwd, float) and np.isnan(eps_fwd)):
                fin_drops.append("G_eps_unknown")
            elif float(eps_fwd) <= 0:
                fin_drops.append("G_eps_nonpositive")
        core_fin_pass = financial and len(fin_drops) == 0

        # ── Trend score (informational only) ─────────────────────────────
        price = row.get("price")
        sma50 = row.get("sma50")
        sma200 = row.get("sma200")
        sma200_prev = row.get("sma200_20d_ago")
        low52w = row.get("low52w")
        high52w = row.get("high52w")
        rs = row.get("rs_percentile")

        trend_score = 0
        if price is not None and sma200 is not None and float(price) > float(sma200):
            trend_score += 1
        if sma50 is not None and sma200 is not None and float(sma50) > float(sma200):
            trend_score += 1
        if (sma200 is not None and sma200_prev is not None
                and not np.isnan(float(sma200_prev))
                and float(sma200) > float(sma200_prev)):
            trend_score += 1
        if price is not None and low52w is not None and float(price) >= float(low52w) * CORE_PRICE_VS_LOW52W:
            trend_score += 1
        if price is not None and high52w is not None and float(price) >= float(high52w) * CORE_PRICE_VS_HIGH52W:
            trend_score += 1
        if rs is not None and not np.isnan(float(rs)) and float(rs) >= CORE_RS_MIN:
            trend_score += 1

        tt_pass = bool(row.get("tt_all_pass", False))
        core_momo_pass = core_pass and tt_pass

        # combine drop reasons
        all_drops = fund_drops + ([f"fin:{r}" for r in fin_drops] if financial else [])

        result_rows.append({
            "code": code,
            "is_financial": financial,
            "core_pass": core_pass,
            "core_fin_pass": core_fin_pass,
            "core_candidate": core_candidate,
            "core_momo_pass": core_momo_pass,
            "trend_score": trend_score,
            "core_drop_reasons": "; ".join(all_drops) if all_drops else None,
        })

    result = pd.DataFrame(result_rows)
    passed = int(result["core_pass"].sum())
    fin_passed = int(result["core_fin_pass"].sum())
    candidates = int(result["core_candidate"].sum())
    momo = int(result["core_momo_pass"].sum())
    n_fin = int(result["is_financial"].sum())
    logger.info(
        "Core screen: %d / %d pass (momo=%d), fin_pass=%d / %d_financial, %d candidates",
        passed, len(result), momo, fin_passed, n_fin, candidates,
    )

    if passed == 0:
        logger.warning("Core: 0件通過 — 閾値またはデータを確認してください")
        if candidates > 0:
            logger.info(
                "Core candidates (coverage不足): %d件 → needs_manual_dividend_check",
                candidates,
            )
    if passed == len(result):
        logger.warning("Core: 全件通過 — 条件が緩すぎる可能性があります")

    return df.merge(result, on="code", how="left")
