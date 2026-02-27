"""Value condition screening and dividend yield calculation.

配当利回りは2種類を算出する:
  dividend_yield_fwd_total : 総額ベース（split-invariant）← 主値
  dividend_yield_fwd       : DPS / price ベース          ← 補助（欠損補完用）

両方取れる場合に乖離が大きい（>50% または <33%）場合は
data_quality_flags に "yield_split_mismatch" を追加し、
コア条件では total ベースを採用する。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HIGH_YIELD_THRESHOLD = 0.06
FCF_YIELD_MIN = 0.04
PER_MAX = 15.0
PBR_MAX = 1.5
PER_SECTOR_PERCENTILE = 0.30  # within-sector bottom 30%

# Mismatch 判定: dps_yield が total_yield の ±50% を超えたら疑わしい
_MISMATCH_UPPER = 1.5
_MISMATCH_LOWER = 0.67


def compute_value_metrics(
    metrics: pd.DataFrame,
    prices: pd.DataFrame | None = None,  # for market cap calculation
) -> pd.DataFrame:
    """Add dividend yield, PER, PBR, FCF yield, value_pass columns.

    metrics must have: code, price, dps_fwd, bps_latest, eps_fwd,
                       fcf_latest, net_shares_latest,
                       total_div_fwd, total_div_actual,
                       data_quality_flags
    Optional sector column for relative PER ranking.
    """
    df = metrics.copy()

    # ── DPS ベース配当利回り（補助 / 欠損補完用） ────────────────────────
    def _yield_dps(row):
        dps = row.get("dps_fwd")
        price = row.get("price")
        if dps is None or price is None or price <= 0:
            return None
        try:
            return float(dps) / float(price)
        except Exception:
            return None

    df["dividend_yield_fwd"] = df.apply(_yield_dps, axis=1)

    # ── 総額ベース配当利回り（split-invariant / 主値） ────────────────────
    def _yield_total(row):
        # ForecastTotal 優先、なければ ResultTotal（＝div_paid 推計値）
        total_div = row.get("total_div_fwd")
        if total_div is None:
            total_div = row.get("total_div_actual")
        price = row.get("price")
        shares = row.get("net_shares_latest")
        if total_div is None or price is None or shares is None:
            return None
        try:
            p, s = float(price), float(shares)
            if p <= 0 or s <= 0:
                return None
            return float(total_div) / (p * s)
        except Exception:
            return None

    df["dividend_yield_fwd_total"] = df.apply(_yield_total, axis=1)

    # ── Split mismatch 検出 ───────────────────────────────────────────────
    def _yield_mismatch(row):
        total = row.get("dividend_yield_fwd_total")
        dps = row.get("dividend_yield_fwd")
        if total is None or dps is None or total <= 0:
            return False
        try:
            ratio = float(dps) / float(total)
            return ratio > _MISMATCH_UPPER or ratio < _MISMATCH_LOWER
        except Exception:
            return False

    df["yield_split_mismatch"] = df.apply(_yield_mismatch, axis=1)

    # ── data_quality_flags に yield_split_mismatch を追記 ─────────────────
    def _update_flags(row):
        val = row.get("data_quality_flags")
        if isinstance(val, list):
            flags = list(val)
        elif val is None or (isinstance(val, float) and np.isnan(val)):
            flags = []
        else:
            flags = [str(val)] if val else []
        if row.get("yield_split_mismatch"):
            if "yield_split_mismatch" not in flags:
                flags.append("yield_split_mismatch")
        return flags

    df["data_quality_flags"] = df.apply(_update_flags, axis=1)

    # high_yield_risk は総額ベースを基準にする（なければ DPS ベース）
    df["high_yield_risk"] = df.apply(
        lambda r: bool(
            (r.get("dividend_yield_fwd_total") or r.get("dividend_yield_fwd") or 0)
            > HIGH_YIELD_THRESHOLD
        ),
        axis=1,
    )

    # ── PER (forward) ────────────────────────────────────────────────────
    def _per(row):
        eps = row.get("eps_fwd")
        price = row.get("price")
        if eps is None or price is None or eps <= 0:
            return None
        try:
            return float(price) / float(eps)
        except Exception:
            return None

    df["per_fwd"] = df.apply(_per, axis=1)

    # ── PBR ──────────────────────────────────────────────────────────────
    def _pbr(row):
        bps = row.get("bps_latest")
        price = row.get("price")
        if bps is None or price is None or bps <= 0:
            return None
        try:
            return float(price) / float(bps)
        except Exception:
            return None

    df["pbr"] = df.apply(_pbr, axis=1)

    # ── FCF yield (FCF per share / price) ────────────────────────────────
    def _fcf_yield(row):
        fcf = row.get("fcf_latest")
        ns = row.get("net_shares_latest")
        price = row.get("price")
        if fcf is None or ns is None or price is None or ns <= 0 or price <= 0:
            return None
        try:
            fcf_per_share = float(fcf) / float(ns)
            return fcf_per_share / float(price)
        except Exception:
            return None

    df["fcf_yield"] = df.apply(_fcf_yield, axis=1)

    # ── Within-sector PER rank ────────────────────────────────────────────
    sector_col = next((c for c in ["Sector33CodeName", "sector33"] if c in df.columns), None)
    if sector_col and "per_fwd" in df.columns:
        df["per_sector_rank"] = df.groupby(sector_col)["per_fwd"].rank(
            pct=True, na_option="keep"
        )
    else:
        df["per_sector_rank"] = None

    # ── value_pass: at least one of PER/PBR/FCF-yield conditions ─────────
    def _value_pass(row):
        per = row.get("per_fwd")
        per_rank = row.get("per_sector_rank")
        pbr = row.get("pbr")
        fcf_y = row.get("fcf_yield")

        cond_per = (per is not None and per <= PER_MAX) or (
            per_rank is not None and per_rank <= PER_SECTOR_PERCENTILE
        )
        cond_pbr = pbr is not None and pbr <= PBR_MAX
        cond_fcf = fcf_y is not None and fcf_y >= FCF_YIELD_MIN

        return bool(cond_per or cond_pbr or cond_fcf)

    df["value_pass"] = df.apply(_value_pass, axis=1)

    pct_pass = df["value_pass"].mean() * 100
    mismatch_n = int(df["yield_split_mismatch"].sum())
    logger.info(
        "value_pass: %.0f%% (%d / %d) | yield_split_mismatch: %d件",
        pct_pass, df["value_pass"].sum(), len(df), mismatch_n,
    )

    return df
