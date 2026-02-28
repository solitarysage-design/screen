"""Compute dividend/cash-flow metrics from fundamentals data.

UNIT CONTRACT: CFO/CFI/FCF/DividendPaid are all TOTAL YEN.
FCF payout = total_div_paid / total_fcf (dimensionally consistent).

non_cut_years_verified は multi-source 年次 DPS（adj_dps）で算出する。
- Source priority: J-Quants FY records → IRBank → yfinance
- 減配判定の許容誤差: 前年比 -1円以内 or -2%以内は「据置」扱い
- coverage_years < 5 の場合: core_pass 対象外だが core_candidate として残す
"""
from __future__ import annotations

import logging
import statistics

import numpy as np
import pandas as pd

from screen.data.dividend_history import get_div_history, AnnualDivRecord

logger = logging.getLogger(__name__)

_LOOKBACK_Y = 5
_PAYOUT_Y = 3
_MIN_FCF_POS_FOR_PAYOUT = 2  # 直近3年でFCF>0が最低2年必要
_NON_CUT_MAX_Y = 10          # non_cut_years の上限
_DIV_HISTORY_SHORT_THRESHOLD = 5  # coverage_years がこれ未満は "短い"

# 減配判定の許容誤差
_CUT_TOL_ABS_JPY = 1.0   # 前年比 -1円以内は据置
_CUT_TOL_REL = 0.02       # 前年比 -2%以内は据置

# non_cut_years_required (Core hard threshold)
NON_CUT_YEARS_REQUIRED = 5


def _compute_adj_dps_jq(
    div_paid: list | None,
    shares_annual: list | None,
) -> list[float | None]:
    """J-Quants の total_div / shares_latest から分割調整済み implied DPS を計算。

    adj_dps_y = total_div_y / shares_latest
              = (total_div_y / shares_y) * (shares_y / shares_latest)
              = dps_y * (shares_y / shares_latest)   ← 現在株数基準への正規化

    total_div は株式分割に無関係に支払総額が記録されるため split-invariant。
    most-recent-first で返す。データ不足の場合は空リスト。
    """
    if not div_paid or not shares_annual:
        return []
    shares_latest = next(
        (float(s) for s in shares_annual if s is not None and float(s) > 0),
        None,
    )
    if not shares_latest:
        return []

    result: list[float | None] = []
    n = min(len(div_paid), len(shares_annual))
    for i in range(n):
        dp = _safe(div_paid[i]) if div_paid[i] is not None else None
        if dp is None or dp <= 0:
            result.append(None)
        else:
            result.append(dp / shares_latest)
    return result


def _safe(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return float(v)


def _is_div_cut(curr: float, prev: float) -> bool:
    """True if curr is a genuine dividend cut (beyond tolerance).

    Tolerance: within 1 JPY or within 2% of previous is treated as unchanged.
    This handles minor adjustments due to rounding / fractional share changes.
    """
    if prev <= 0:
        return False  # zero or negative prev: no cut judgment possible
    diff = prev - curr  # positive = cut, negative = raise
    if diff <= 0:
        return False  # raised or flat
    # Check tolerance
    rel_diff = diff / prev
    if diff <= _CUT_TOL_ABS_JPY or rel_diff <= _CUT_TOL_REL:
        return False  # within tolerance: treated as unchanged
    return True


def _non_cut_years_from_records(
    records: list[AnnualDivRecord],
) -> tuple[float | None, int]:
    """Compute non_cut_years_verified and coverage_years from AnnualDivRecord list.

    Returns:
        (non_cut_years_verified, coverage_years)
        - non_cut_years_verified: consecutive years from the most recent
          where no genuine cut was detected.
        - coverage_years: total number of annual records available.
    """
    # Sort descending by year (most-recent first)
    sorted_records = sorted(records, key=lambda r: r.year, reverse=True)
    coverage = len(sorted_records)

    if coverage < 2:
        return None, coverage

    count = 0
    for i in range(len(sorted_records) - 1):
        curr_dps = _safe(sorted_records[i].dps)
        prev_dps = _safe(sorted_records[i + 1].dps)
        if curr_dps is None or prev_dps is None:
            break
        if _is_div_cut(curr_dps, prev_dps):
            break
        count += 1

    return float(min(count, _NON_CUT_MAX_Y)), coverage


def _non_cut_years_adj(
    dps_list: list | None,
    shares_list: list | None,
) -> tuple[float | None, int]:
    """Legacy: split-adjusted DPS list → non_cut_years (fallback path)."""
    if not dps_list:
        return None, 0

    shares_latest = None
    if shares_list:
        shares_latest = _safe(shares_list[0])

    adj: list[float | None] = []
    for i, dps in enumerate(dps_list):
        d = _safe(dps)
        if d is None:
            adj.append(None)
            continue
        if shares_latest and shares_latest > 0 and shares_list and i < len(shares_list):
            shares_y = _safe(shares_list[i])
            if shares_y and shares_y > 0:
                adj.append(d * (shares_y / shares_latest))
                continue
        adj.append(d)

    coverage = sum(1 for v in adj if v is not None)
    if coverage < 2:
        return None, coverage

    count = 0
    for i in range(len(adj) - 1):
        curr, prev = _safe(adj[i]), _safe(adj[i + 1])
        if curr is None or prev is None:
            break
        if _is_div_cut(curr, prev):
            break
        count += 1
    return float(min(count, _NON_CUT_MAX_Y)), coverage


def _pos_ratio(vals: list | None, n: int) -> float | None:
    if not vals:
        return None
    subset = [_safe(v) for v in vals[:n] if _safe(v) is not None]
    if not subset:
        return None
    return sum(1 for v in subset if v > 0) / len(subset)


def _fcf_payout_3y(
    fcf_list: list | None,
    div_paid_list: list | None,
) -> tuple[float | None, bool]:
    """Cumulative 3-year FCF payout ratio.

    Returns (ratio, hard_fail_flag).
    hard_fail=True when < 2 FY with FCF>0 in last 3 years.

    Formula: sum(div_paid over 3FY) / sum(FCF for FY where FCF>0 in 3FY)
    """
    if not fcf_list or not div_paid_list:
        return None, False

    n = min(_PAYOUT_Y, len(fcf_list), len(div_paid_list))
    if n == 0:
        return None, False

    total_fcf_pos = 0.0
    total_div = 0.0
    fcf_pos_count = 0

    for i in range(n):
        fcf = _safe(fcf_list[i])
        div = _safe(div_paid_list[i])
        if fcf is None or div is None:
            continue
        total_div += div
        if fcf > 0:
            total_fcf_pos += fcf
            fcf_pos_count += 1

    hard_fail = fcf_pos_count < _MIN_FCF_POS_FOR_PAYOUT

    if total_fcf_pos <= 0:
        return None, True  # hard fail: no positive FCF

    return total_div / total_fcf_pos, hard_fail


def compute_fundamentals_metrics(fund_df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics from get_fundamentals() output.

    Input cols: code, fcf_annual, div_paid_annual, dps_actual_annual,
                cfo_annual, shares_annual, jq_fy_dates,
                total_div_fwd, total_div_actual,
                dps_fwd, bps_latest, eps_fwd, net_shares_latest, ...

    Output adds (per code):
        non_cut_years, non_cut_years_verified, non_cut_years_required,
        coverage_years, div_source_used, needs_manual_dividend_check,
        cfo_pos_5y_ratio, fcf_pos_5y_ratio,
        fcf_payout_3y, fcf_payout_hard_fail,
        fcf_latest, div_paid_latest,
        total_div_fwd, total_div_actual (passthrough),
        dps_fwd (passthrough), bps_latest (passthrough), eps_fwd (passthrough),
        metrics_coverage, data_quality_flags (merged)
    """
    rows = []

    all_payouts = []  # for post-validation

    for _, r in fund_df.iterrows():
        code = str(r.get("code", ""))
        fcf = r.get("fcf_annual")
        div_paid = r.get("div_paid_annual")
        cfo = r.get("cfo_annual")
        dps_actual = r.get("dps_actual_annual")
        shares_annual = r.get("shares_annual")
        jq_fy_dates = r.get("jq_fy_dates") or []
        # Extract fiscal year-end month from the most recent FY date
        fiscal_month: int | None = None
        if jq_fy_dates:
            try:
                fiscal_month = pd.to_datetime(jq_fy_dates[0]).month
            except Exception:
                pass
        quality_flags: list[str] = list(r.get("data_quality_flags") or [])

        # ── A) Dividend history: multi-source ────────────────────────────
        # total_div / shares_latest で split-adjusted implied DPS を計算（優先）
        # raw DPS（ResultDividendPerShareAnnual）は分割前後で数値が変わるため回避
        jq_adj_dps = _compute_adj_dps_jq(div_paid, shares_annual)
        div_hist = get_div_history(
            code=code,
            jq_dps_list=jq_adj_dps if jq_adj_dps else dps_actual,
            jq_fy_dates=jq_fy_dates,
        )

        div_source_used: str = div_hist.source_used
        coverage_years: int = div_hist.coverage_years

        if div_hist.annual_records:
            non_cut, _ = _non_cut_years_from_records(div_hist.annual_records)
        else:
            # Fallback: use legacy share-adjusted calculation
            non_cut, coverage_years = _non_cut_years_adj(dps_actual, shares_annual)
            div_source_used = "jquants"

        non_cut_years_verified = non_cut
        needs_manual_dividend_check = coverage_years < NON_CUT_YEARS_REQUIRED

        # Flag split states
        if div_hist.has_splits and div_hist.split_adjusted:
            quality_flags.append("div_split_adjusted")
        elif div_hist.has_splits and not div_hist.split_adjusted:
            quality_flags.append("div_has_splits_unadjusted")

        # 配当履歴が短い場合はフラグを追加
        if coverage_years < _DIV_HISTORY_SHORT_THRESHOLD:
            quality_flags.append("div_history_short")

        # ── B) CFO positive ratio 5y ─────────────────────────────────────
        cfo_pos = _pos_ratio(cfo, _LOOKBACK_Y)

        # ── C) FCF positive ratio 5y ─────────────────────────────────────
        fcf_pos = _pos_ratio(fcf, _LOOKBACK_Y)

        # ── D) FCF payout 3y cumulative ──────────────────────────────────
        fcf_payout_3y, fcf_hard_fail = _fcf_payout_3y(fcf, div_paid)

        # Latest-year point values
        fcf_latest = _safe(fcf[0]) if fcf else None
        div_paid_latest = _safe(div_paid[0]) if div_paid else None

        if fcf_payout_3y is not None:
            all_payouts.append((code, fcf_payout_3y))

        # Sanity: negative or absurdly large payout
        if fcf_payout_3y is not None:
            if fcf_payout_3y < 0:
                quality_flags.append("fcf_payout_negative")
            elif fcf_payout_3y > 10:
                quality_flags.append(f"fcf_payout_extreme:{fcf_payout_3y:.2f}")

        metric_vals = [non_cut, cfo_pos, fcf_pos, fcf_payout_3y]
        cov = sum(1 for v in metric_vals if v is not None) / len(metric_vals)

        rows.append({
            "code": code,
            # Legacy compat alias
            "non_cut_years": non_cut,
            # New columns (req 5)
            "non_cut_years_verified": non_cut_years_verified,
            "non_cut_years_required": NON_CUT_YEARS_REQUIRED,
            "coverage_years": coverage_years,
            "div_source_used": div_source_used,
            "needs_manual_dividend_check": needs_manual_dividend_check,
            # CFO/FCF metrics
            "cfo_pos_5y_ratio": cfo_pos,
            "fcf_pos_5y_ratio": fcf_pos,
            "fcf_payout_3y": fcf_payout_3y,
            "fcf_payout_hard_fail": bool(fcf_hard_fail),
            "fcf_latest": fcf_latest,
            "div_paid_latest": div_paid_latest,
            # Passthroughs
            "total_div_fwd": _safe(r.get("total_div_fwd")),
            "total_div_actual": _safe(r.get("total_div_actual")),
            "dps_fwd": _safe(r.get("dps_fwd")),
            "bps_latest": _safe(r.get("bps_latest")),
            "eps_fwd": _safe(r.get("eps_fwd")),
            "net_shares_latest": _safe(r.get("net_shares_latest")),
            "fiscal_month": fiscal_month,
            "eps_q_list": r.get("eps_q_list"),
            "metrics_coverage": round(cov, 2),
            "data_quality_flags": quality_flags,
        })

    result = pd.DataFrame(rows)

    # ── Post-validation: fcf_payout_3y median check ──────────────────────
    if all_payouts:
        payout_vals = [v for _, v in all_payouts]
        med = statistics.median(payout_vals)
        logger.info(
            "fcf_payout_3y stats: n=%d median=%.4f mean=%.4f min=%.4f max=%.4f",
            len(payout_vals), med,
            sum(payout_vals) / len(payout_vals),
            min(payout_vals), max(payout_vals),
        )
        if med < 1e-4:
            raise RuntimeError(
                f"fcf_payout_3y median={med:.2e} — 単位ズレ/実装バグの疑い。"
                "DPS×株数 と FCF の単位を確認してください。"
            )
        # Log top 10
        top10 = sorted(all_payouts, key=lambda x: x[1], reverse=True)[:10]
        logger.info("fcf_payout_3y 上位10件: %s", [(c, f"{v:.3f}") for c, v in top10])

    # ── Post-validation: coverage summary ────────────────────────────────
    if not result.empty and "coverage_years" in result.columns:
        cov_series = result["coverage_years"]
        n_short = int((cov_series < NON_CUT_YEARS_REQUIRED).sum())
        src_counts = result["div_source_used"].value_counts().to_dict()
        logger.info(
            "Dividend coverage: n_short=%d/%d sources=%s",
            n_short, len(result), src_counts,
        )

    return result
