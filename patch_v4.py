"""Standalone patch → candidates_fixed_v4.csv

Changes from v3:
  A) J-Quants auth preflight → SATELLITE_ONLY mode if auth unavailable
  B) Holdings statements re-fetch when auth OK
  C) yield_total consistency: explicit yield_used column, DPS fallback fills
     dividend_yield_fwd_total before core logic runs (not inside it)
  D) Core pass logic uses ONLY dividend_yield_fwd_total — no implicit dps fallback
  E) Assertions: yield_total NaN & core_pass = 0, in_holdings=21, drop_reason NaN=0
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
HOLDINGS_CSV = Path("C:/Users/solit/Downloads/holdings_extracted_20260224.csv")
INPUT_CSV    = Path("C:/Users/solit/projects/screen/output/candidates_fixed_v3.csv")
OUT_DIR      = Path("C:/Users/solit/projects/screen/output")

# ── Constants (mirror core_screen.py / config.py) ─────────────────────────────
FINANCIAL_SECTOR33 = {"銀行業", "保険業", "その他金融業", "証券、商品先物取引業"}
CYCLICAL_SECTOR33  = {
    "鉄鋼", "非鉄金属", "石油・石炭製品", "化学", "海運業", "空運業", "鉱業", "建設業",
}

CORE_NON_CUT_MIN   = 5
CORE_YIELD_MIN     = 0.03
CORE_CFO_POS_MIN   = 0.80
CORE_FCF_POS_MIN   = 0.60
CORE_FCF_PAYOUT_NC = 0.70
CORE_FCF_PAYOUT_CY = 0.60

# Columns whose NaN → data_missing:key in drop_reason
_CORE_REQUIRED_COLS: list[tuple[str, str]] = [
    ("non_cut_years_verified",   "non_cut_years"),
    ("dividend_yield_fwd_total", "yield_total"),
    ("cfo_pos_5y_ratio",         "cfo_pos_ratio"),
    ("fcf_pos_5y_ratio",         "fcf_pos_ratio"),
    ("fcf_payout_3y",            "fcf_payout"),
]

STAGE_B_YIELD_THRESHOLD = 0.02  # include in Stage B if yield_total >= 2%


# ── Helpers ───────────────────────────────────────────────────────────────────
def _nan(v) -> bool:
    if v is None:
        return True
    if v is pd.NA:
        return True
    try:
        return bool(np.isnan(float(v)))
    except (TypeError, ValueError):
        return False


def _val(v) -> float | None:
    if _nan(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_list_col(v) -> list[str]:
    """Parse a column that may be a list, str, or NaN."""
    if isinstance(v, list):
        return v
    if _nan(v) or str(v).strip() in ("", "nan", "[]"):
        return []
    return [x.strip() for x in str(v).split(";") if x.strip()]


# ── A) J-Quants auth preflight ────────────────────────────────────────────────
def check_jquants_auth() -> tuple[bool, str]:
    """Return (auth_ok, error_message).

    Checks env vars first; if present, attempts a lightweight API call to
    confirm the credentials actually work.

    Supported env vars (checked in order of priority):
        JQUANTS_EMAIL / JQUANTS_API_MAIL_ADDRESS / JQUANTS_MAIL_ADDRESS
        JQUANTS_PASSWORD / JQUANTS_API_PASSWORD
        JQUANTS_REFRESH_TOKEN / JQUANTS_API_REFRESH_TOKEN
    """
    email = (
        os.environ.get("JQUANTS_EMAIL")
        or os.environ.get("JQUANTS_API_MAIL_ADDRESS")
        or os.environ.get("JQUANTS_MAIL_ADDRESS")
        or ""
    ).strip()
    password = (
        os.environ.get("JQUANTS_PASSWORD")
        or os.environ.get("JQUANTS_API_PASSWORD")
        or ""
    ).strip()
    refresh_token = (
        os.environ.get("JQUANTS_REFRESH_TOKEN")
        or os.environ.get("JQUANTS_API_REFRESH_TOKEN")
        or ""
    ).strip()

    has_creds = bool((email and password) or refresh_token)
    if not has_creds:
        return False, (
            "no_jquants_auth（環境変数が未設定: "
            "JQUANTS_EMAIL + JQUANTS_PASSWORD または JQUANTS_REFRESH_TOKEN が必要）"
        )

    # Attempt a real auth call to confirm credentials work
    try:
        import jquantsapi as jq  # type: ignore
        if refresh_token:
            client = jq.Client(refresh_token=refresh_token)
        else:
            client = jq.Client(mail_address=email, password=password)
        # Minimal test: fetch statements for one small company
        _ = client.get_fins_statements(code="86970")
        return True, ""
    except ImportError:
        return False, "no_jquants_auth（jquantsapi パッケージが未インストール）"
    except Exception as exc:
        return False, f"no_jquants_auth（認証失敗: {str(exc)[:100]}）"


# ── B) Holdings statements fetch ──────────────────────────────────────────────
def fetch_holdings_fundamentals(
    holding_codes: list[str],
    prices: dict[str, float],
) -> pd.DataFrame | None:
    """Fetch J-Quants statements + compute metrics for holding stocks.

    Returns a DataFrame with per-code fundamentals metrics merged with
    yield calculations. Returns None on failure.
    """
    logger.info("Fetching J-Quants statements for %d holdings...", len(holding_codes))
    try:
        from screen.data.fundamentals import get_fundamentals
        from screen.features.fundamentals_metrics import compute_fundamentals_metrics

        raw_fund = get_fundamentals(holding_codes)
        if raw_fund.empty:
            logger.warning("get_fundamentals returned empty DataFrame for holdings")
            return None

        metrics = compute_fundamentals_metrics(raw_fund)

        # Add price column for yield computation
        metrics["price"] = metrics["code"].map(prices)

        # Compute yield_total = total_div / (price * net_shares)
        def _yield_total(row):
            total_div = _val(row.get("total_div_fwd")) or _val(row.get("total_div_actual"))
            price     = _val(row.get("price"))
            shares    = _val(row.get("net_shares_latest"))
            if total_div is None or price is None or shares is None:
                return None
            if price <= 0 or shares <= 0:
                return None
            return total_div / (price * shares)

        metrics["dividend_yield_fwd_total_jq"] = metrics.apply(_yield_total, axis=1)

        # DPS yield
        def _yield_dps(row):
            dps   = _val(row.get("dps_fwd"))
            price = _val(row.get("price"))
            if dps is None or price is None or price <= 0:
                return None
            return dps / price

        metrics["dividend_yield_fwd_jq"] = metrics.apply(_yield_dps, axis=1)

        logger.info(
            "Holdings fundamentals fetched: %d rows, yield_total non-NaN=%d",
            len(metrics),
            int(metrics["dividend_yield_fwd_total_jq"].notna().sum()),
        )
        return metrics

    except Exception as exc:
        logger.error("Holdings fundamentals fetch failed: %s", exc, exc_info=True)
        return None


# ── Stage A: yield_total via DPS fallback ─────────────────────────────────────
def apply_stage_a_yield_total(df: pd.DataFrame) -> pd.DataFrame:
    """Fill dividend_yield_fwd_total using DPS fallback where still missing.

    Priority:
      1) Already have yield_total from J-Quants (yield_used = "total")
      2) Have dividend_yield_fwd (DPS/price) → copy as fallback
         (yield_used = "dps_fallback", quality flag "yield_total_fallback_dps")
      3) No yield at all (yield_used = "none")
    """
    if "yield_used" not in df.columns:
        df["yield_used"] = pd.NA

    n_total    = 0
    n_fallback = 0
    n_none     = 0

    for idx, row in df.iterrows():
        yield_total = _val(row.get("dividend_yield_fwd_total"))
        yield_dps   = _val(row.get("dividend_yield_fwd"))

        if not _nan(yield_total):
            # Already have proper yield_total — always label as "total"
            df.at[idx, "yield_used"] = "total"
            n_total += 1
        elif yield_dps is not None and yield_dps > 0:
            # Fallback: use DPS yield as proxy for total yield
            df.at[idx, "dividend_yield_fwd_total"] = yield_dps
            df.at[idx, "yield_used"] = "dps_fallback"
            # Add quality flag
            qf = _parse_list_col(row.get("data_quality_flags"))
            if "yield_total_fallback_dps" not in qf:
                qf.append("yield_total_fallback_dps")
            df.at[idx, "data_quality_flags"] = qf
            n_fallback += 1
        else:
            df.at[idx, "yield_used"] = "none"
            n_none += 1

    logger.info(
        "Stage A yield_total: total=%d  dps_fallback=%d  none=%d",
        n_total, n_fallback, n_none,
    )
    return df


# ── Core pass recompute (v4: yield_total only, no implicit fallback) ───────────
def _recompute_core_pass_v4(row: pd.Series) -> tuple[bool, bool, bool, str]:
    """Returns (core_pass, core_fin_pass, core_candidate, drop_reasons_str).

    IMPORTANT: Uses ONLY dividend_yield_fwd_total.
    If yield_total is NaN → E_div_yield fails → core_pass = False.
    """
    sector    = str(row.get("Sector33CodeName") or "")
    financial = sector in FINANCIAL_SECTOR33
    cyclical  = sector in CYCLICAL_SECTOR33
    payout_cap = CORE_FCF_PAYOUT_CY if cyclical else CORE_FCF_PAYOUT_NC

    non_cut = _val(row.get("non_cut_years_verified")) or _val(row.get("non_cut_years"))
    cov     = int(_val(row.get("coverage_years")) or 0)
    cov_insuf = cov < CORE_NON_CUT_MIN

    fund_drops: list[str] = []

    # A: non-cut years
    if non_cut is None or non_cut < CORE_NON_CUT_MIN:
        fund_drops.append("A_non_cut_years")
    if cov_insuf and "A_non_cut_years" not in fund_drops:
        fund_drops.append("A_coverage_insufficient")

    # B/C/D: CFO/FCF (non-financial only)
    if not financial:
        cfo = _val(row.get("cfo_pos_5y_ratio"))
        if cfo is None or cfo < CORE_CFO_POS_MIN:
            fund_drops.append("B_cfo_pos_ratio")
        fcf_pos = _val(row.get("fcf_pos_5y_ratio"))
        if fcf_pos is None or fcf_pos < CORE_FCF_POS_MIN:
            fund_drops.append("C_fcf_pos_ratio")
        payout = _val(row.get("fcf_payout_3y"))
        if payout is None or payout > payout_cap:
            fund_drops.append("D_fcf_payout")
        if row.get("fcf_payout_hard_fail") is True:
            fund_drops.append("D_fcf_hard_fail")
    else:
        fund_drops.append("fin_sector_excluded_from_core")

    # E: yield_total ONLY — no dps fallback here (done in Stage A pre-processing)
    yield_total = _val(row.get("dividend_yield_fwd_total"))
    if yield_total is None or yield_total < CORE_YIELD_MIN:
        fund_drops.append("E_div_yield")

    # F: value_pass
    if not row.get("value_pass", False):
        fund_drops.append("F_value_pass")

    core_pass = (not financial) and len(fund_drops) == 0

    non_cov_drops = [
        d for d in fund_drops
        if d not in ("A_non_cut_years", "A_coverage_insufficient",
                     "fin_sector_excluded_from_core")
    ]
    core_candidate = (
        not core_pass and not financial and cov_insuf and len(non_cov_drops) == 0
    )

    # core_fin_pass: financial sector alternative
    fin_drops: list[str] = []
    if financial:
        if non_cut is None or non_cut < CORE_NON_CUT_MIN:
            fin_drops.append("A_non_cut_years")
        if cov_insuf and "A_non_cut_years" not in fin_drops:
            fin_drops.append("A_coverage_insufficient")
        if yield_total is None or yield_total < CORE_YIELD_MIN:
            fin_drops.append("E_div_yield")
        if not row.get("value_pass", False):
            fin_drops.append("F_value_pass")
        eps = _val(row.get("eps_fwd"))
        if eps is None:
            fin_drops.append("G_eps_unknown")
        elif eps <= 0:
            fin_drops.append("G_eps_nonpositive")

    core_fin_pass = financial and len(fin_drops) == 0

    all_drops = fund_drops + ([f"fin:{r}" for r in fin_drops] if financial else [])
    return core_pass, core_fin_pass, core_candidate, "; ".join(all_drops) if all_drops else ""


# ── Drop reason builder ───────────────────────────────────────────────────────
def _make_drop_reason(row: pd.Series, satellite_only: bool = False) -> str:
    """Unified drop_reason. Never NaN, never empty string."""
    reasons: list[str] = []

    if satellite_only:
        reasons.append("no_jquants_auth_core_disabled")

    for field in ["core_drop_reasons", "satellite_drop_reasons"]:
        dr = row.get(field)
        if dr is None or (isinstance(dr, float) and np.isnan(dr)):
            continue
        for r in str(dr).split(";"):
            r = r.strip()
            if r and r.lower() != "nan" and r not in reasons:
                reasons.append(r)

    # data_missing for NaN core columns
    for col, key in _CORE_REQUIRED_COLS:
        if _nan(row.get(col)):
            tag = f"data_missing:{key}"
            if tag not in reasons:
                reasons.append(tag)

    # error_flags
    ef = row.get("error_flags")
    for f in (_parse_list_col(ef) if not isinstance(ef, list) else ef):
        if f and f not in reasons:
            reasons.append(f)

    return "; ".join(reasons) if reasons else "OK"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── A) Auth preflight ─────────────────────────────────────────────────────
    logger.info("=== J-Quants auth preflight ===")
    auth_ok, auth_err = check_jquants_auth()
    if auth_ok:
        logger.info("J-Quants auth: OK — core mode enabled")
        SATELLITE_ONLY = False
    else:
        logger.warning("J-Quants auth: FAILED — satellite_only mode")
        logger.warning("  理由: %s", auth_err)
        logger.warning(
            "  core_pass は全件 False になります。"
            "core 判定を有効にするには環境変数を設定してください: "
            "JQUANTS_EMAIL + JQUANTS_PASSWORD または JQUANTS_REFRESH_TOKEN"
        )
        SATELLITE_ONLY = True

    # ── Load inputs ───────────────────────────────────────────────────────────
    h_df = pd.read_csv(HOLDINGS_CSV, dtype=str)
    holdings_codes: set[str] = set(h_df["code_jquants_5digit"].str.strip().dropna())
    assert len(holdings_codes) == 21, f"Expected 21 holdings, got {len(holdings_codes)}"
    logger.info("Holdings: %d codes", len(holdings_codes))

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    df["code"] = df["code"].astype(str).str.strip()
    logger.info("Loaded v3 CSV: %d rows, %d cols", len(df), len(df.columns))

    # ── Fix in_holdings ───────────────────────────────────────────────────────
    df["in_holdings"] = df["code"].isin(holdings_codes)
    assert int(df["in_holdings"].sum()) == 21
    logger.info("in_holdings: %d True ✓", int(df["in_holdings"].sum()))

    # ── Parse list columns ────────────────────────────────────────────────────
    for col in ["error_flags", "data_quality_flags"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_col)

    # ── is_financial ──────────────────────────────────────────────────────────
    df["is_financial"] = df["Sector33CodeName"].isin(FINANCIAL_SECTOR33)

    # ── B) Holdings J-Quants re-fetch (auth_ok only) ──────────────────────────
    if auth_ok:
        prices_dict = dict(zip(df["code"], df["price"].apply(_val)))
        h_codes = sorted(holdings_codes)
        holdings_metrics = fetch_holdings_fundamentals(h_codes, prices_dict)

        if holdings_metrics is not None:
            # Update main DataFrame with fresh fundamentals for holdings
            update_cols = [
                "non_cut_years_verified", "coverage_years",
                "cfo_pos_5y_ratio", "fcf_pos_5y_ratio",
                "fcf_payout_3y", "fcf_payout_hard_fail",
                "net_shares_latest", "dps_fwd", "eps_fwd",
                "total_div_fwd", "total_div_actual",
            ]
            jq_yield_cols = ["dividend_yield_fwd_total_jq", "dividend_yield_fwd_jq"]

            for _, fm_row in holdings_metrics.iterrows():
                code = str(fm_row.get("code", "")).strip()
                mask = df["code"] == code
                idx_list = df.index[mask].tolist()
                if not idx_list:
                    continue
                idx = idx_list[0]

                for col in update_cols:
                    if col in holdings_metrics.columns:
                        df.at[idx, col] = fm_row.get(col)

                # Update yield columns
                yq_total = _val(fm_row.get("dividend_yield_fwd_total_jq"))
                yq_dps   = _val(fm_row.get("dividend_yield_fwd_jq"))
                if yq_total is not None:
                    df.at[idx, "dividend_yield_fwd_total"] = yq_total
                    df.at[idx, "yield_used"] = "total"
                if yq_dps is not None:
                    df.at[idx, "dividend_yield_fwd"] = yq_dps

                # Merge error_flags; remove stale no_auth flag on success
                ef = _parse_list_col(df.at[idx, "error_flags"])
                ef = [f for f in ef if f != "stmt_fetch_failed:no_jquants_auth"]
                new_ef = _parse_list_col(fm_row.get("error_flags") or [])
                for f in new_ef:
                    if f not in ef:
                        ef.append(f)
                df.at[idx, "error_flags"] = ef

            logger.info("Holdings: updated %d rows from J-Quants", len(holdings_metrics))

            # Stage B: fetch J-Quants for yield_total >= 2% ∪ in_holdings
            stage_b_mask = (
                (df["dividend_yield_fwd_total"].fillna(0) >= STAGE_B_YIELD_THRESHOLD)
                | df["in_holdings"]
            ) & ~df["is_financial"]
            stage_b_codes = df.loc[stage_b_mask, "code"].tolist()
            # Exclude holdings (already fetched above)
            stage_b_codes = [c for c in stage_b_codes if c not in holdings_codes]
            logger.info("Stage B: fetching J-Quants for %d candidates...", len(stage_b_codes))

            if stage_b_codes:
                try:
                    from screen.data.fundamentals import get_fundamentals
                    from screen.features.fundamentals_metrics import compute_fundamentals_metrics

                    sb_raw     = get_fundamentals(stage_b_codes)
                    sb_metrics = compute_fundamentals_metrics(sb_raw)
                    sb_metrics["price"] = sb_metrics["code"].map(prices_dict)

                    def _y_total(r):
                        td = _val(r.get("total_div_fwd")) or _val(r.get("total_div_actual"))
                        p  = _val(r.get("price"))
                        s  = _val(r.get("net_shares_latest"))
                        return (td / (p * s)) if (td and p and p > 0 and s and s > 0) else None

                    sb_metrics["dividend_yield_fwd_total_jq"] = sb_metrics.apply(_y_total, axis=1)

                    sb_update_cols = update_cols + ["dividend_yield_fwd_total_jq"]
                    for _, sb_row in sb_metrics.iterrows():
                        code = str(sb_row.get("code", "")).strip()
                        mask = df["code"] == code
                        idx_list = df.index[mask].tolist()
                        if not idx_list:
                            continue
                        idx = idx_list[0]
                        for col in update_cols:
                            if col in sb_metrics.columns:
                                df.at[idx, col] = sb_row.get(col)
                        yq_total = _val(sb_row.get("dividend_yield_fwd_total_jq"))
                        if yq_total is not None:
                            df.at[idx, "dividend_yield_fwd_total"] = yq_total
                            df.at[idx, "yield_used"] = "total"

                    logger.info("Stage B: updated %d rows", len(sb_metrics))
                except Exception as exc:
                    logger.error("Stage B J-Quants fetch failed: %s", exc, exc_info=True)
        else:
            # holdings fetch failed — tag error_flags
            for code in holdings_codes:
                mask = df["code"] == code
                idx_list = df.index[mask].tolist()
                if not idx_list:
                    continue
                idx = idx_list[0]
                ef = _parse_list_col(df.at[idx, "error_flags"])
                tag = "stmt_fetch_failed:holdings_fetch_error"
                if tag not in ef:
                    ef.append(tag)
                df.at[idx, "error_flags"] = ef

    else:
        # satellite_only: mark holdings that are missing J-Quants data
        h_missing = df["in_holdings"] & df["cfo_pos_5y_ratio"].isna()
        for idx in df.index[h_missing]:
            ef = _parse_list_col(df.at[idx, "error_flags"])
            tag = "stmt_fetch_failed:no_jquants_auth"
            if tag not in ef:
                ef.append(tag)
            df.at[idx, "error_flags"] = ef
        logger.info(
            "satellite_only: %d holdings lack J-Quants CFO/FCF (stmt_fetch_failed:no_jquants_auth)",
            int(h_missing.sum()),
        )

    # ── Stage A: yield_total via DPS fallback ─────────────────────────────────
    logger.info("Stage A: filling yield_total via DPS fallback...")
    before_nn = int(df["dividend_yield_fwd_total"].notna().sum())
    df = apply_stage_a_yield_total(df)
    after_nn  = int(df["dividend_yield_fwd_total"].notna().sum())
    logger.info(
        "Stage A: dividend_yield_fwd_total non-NaN: %d → %d (+ %d)",
        before_nn, after_nn, after_nn - before_nn,
    )
    logger.info("yield_used distribution: %s", df["yield_used"].value_counts().to_dict())

    # ── Core pass recompute ───────────────────────────────────────────────────
    logger.info("Recomputing core_pass for all %d rows...", len(df))
    core_results = df.apply(_recompute_core_pass_v4, axis=1)
    df["core_pass"]         = [r[0] for r in core_results]
    df["core_fin_pass"]     = [r[1] for r in core_results]
    df["core_candidate"]    = [r[2] for r in core_results]
    df["core_drop_reasons"] = [r[3] for r in core_results]
    df["core_momo_pass"]    = df["core_pass"] & df.get("tt_all_pass", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    # ── Satellite_only override ───────────────────────────────────────────────
    if SATELLITE_ONLY:
        logger.warning(
            "satellite_only mode: forcing core_pass=False for all %d rows",
            int(df["core_pass"].sum()),
        )
        df["core_pass"]      = False
        df["core_fin_pass"]  = False
        df["core_momo_pass"] = False

    # ── Drop reason ───────────────────────────────────────────────────────────
    df["drop_reason"] = df.apply(
        lambda r: _make_drop_reason(r, satellite_only=SATELLITE_ONLY), axis=1
    )

    # ── E) Validation ─────────────────────────────────────────────────────────
    logger.info("=== Validation ===")

    n_in_holdings = int(df["in_holdings"].sum())
    assert n_in_holdings == 21, f"FAIL: in_holdings={n_in_holdings}"
    logger.info("in_holdings True: %d ✓", n_in_holdings)

    # Holdings column NaN check
    h_mask = df["in_holdings"]
    check_cols = [
        ("dividend_yield_fwd_total", "yield_total"),
        ("dividend_yield_fwd",       "yield_dps"),
        ("non_cut_years_verified",   "non_cut_years"),
        ("cfo_pos_5y_ratio",         "cfo_pos_ratio"),
        ("fcf_pos_5y_ratio",         "fcf_pos_ratio"),
        ("fcf_payout_3y",            "fcf_payout"),
    ]
    for col, key in check_cols:
        if col in df.columns:
            n_nan = int(df.loc[h_mask, col].isna().sum())
            status = "OK" if n_nan == 0 else "WARNING"
            logger.info("[%s] 保有株 %s NaN: %d / 21", status, key, n_nan)

    # yield_total coverage
    yt_nn = int(df["dividend_yield_fwd_total"].notna().sum())
    logger.info(
        "dividend_yield_fwd_total non-NaN: %d / %d (v3 was 143)",
        yt_nn, len(df),
    )

    # C) CRITICAL: yield_total NaN & core_pass/core_fin_pass must be 0
    yield_nan = df["dividend_yield_fwd_total"].isna()
    cp_bug  = df[yield_nan & df["core_pass"]].shape[0]
    cfp_bug = df[yield_nan & df["core_fin_pass"]].shape[0]
    assert cp_bug  == 0, f"FAIL: yield_total NaN & core_pass=True: {cp_bug} rows"
    assert cfp_bug == 0, f"FAIL: yield_total NaN & core_fin_pass=True: {cfp_bug} rows"
    logger.info("yield_total NaN & core_pass=True: 0 ✓")
    logger.info("yield_total NaN & core_fin_pass=True: 0 ✓")

    # Financial stocks in core_pass
    fin_in_core = int((df["is_financial"] & df["core_pass"]).sum())
    assert fin_in_core == 0, f"FAIL: {fin_in_core} financial in core_pass"
    logger.info("Financial stocks in core_pass: 0 ✓")

    # drop_reason NaN
    n_dr_nan = int(df["drop_reason"].isna().sum())
    assert n_dr_nan == 0, f"FAIL: drop_reason NaN={n_dr_nan}"
    logger.info("drop_reason NaN: 0 ✓")

    # Summary counts
    n_core    = int(df["core_pass"].sum())
    n_fin_p   = int(df["core_fin_pass"].sum())
    n_sat     = int(df["satellite_pass"].sum()) if "satellite_pass" in df.columns else 0
    n_cand    = int(df["core_candidate"].sum())
    logger.info(
        "core_pass=%d  core_fin_pass=%d  core_candidate=%d  satellite_pass=%d",
        n_core, n_fin_p, n_cand, n_sat,
    )

    # data_missing top
    dm = df["drop_reason"].str.split("; ").explode().str.strip()
    dm_counts = dm[dm.str.startswith("data_missing:")].value_counts().head(8)
    if not dm_counts.empty:
        logger.info("data_missing top:\n%s", dm_counts.to_string())

    sf_counts = dm[dm.str.startswith("stmt_fetch_failed:")].value_counts().head(5)
    if not sf_counts.empty:
        logger.info("stmt_fetch_failed:\n%s", sf_counts.to_string())

    mode_str = "satellite_only (no_jquants_auth)" if SATELLITE_ONLY else "full (jquants_ok)"
    logger.info("mode: %s", mode_str)

    # ── Stringify list columns for CSV ────────────────────────────────────────
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(3)
            if not sample.empty and any(isinstance(v, list) for v in sample):
                df[col] = df[col].apply(
                    lambda v: "; ".join(str(x) for x in v) if isinstance(v, list) else (v or "")
                )

    # Ensure required output columns exist
    for col in ["core_fin_pass", "is_financial", "error_flags", "yield_used"]:
        if col not in df.columns:
            df[col] = ""

    # ── Save candidates_fixed_v4.csv ──────────────────────────────────────────
    out_v4 = OUT_DIR / "candidates_fixed_v4.csv"
    df.to_csv(out_v4, index=False, encoding="utf-8-sig")
    logger.info("candidates_fixed_v4.csv → %s (%d rows)", out_v4, len(df))

    # ── Save holdings_debug_fixed_v4.csv ─────────────────────────────────────
    debug_cols = [
        "code", "in_holdings", "is_financial",
        "core_pass", "core_fin_pass", "core_candidate", "core_momo_pass",
        "satellite_pass", "drop_reason", "core_drop_reasons",
        "yield_used",
        "dividend_yield_fwd_total", "dividend_yield_fwd",
        "yield_split_mismatch", "non_cut_years_verified", "coverage_years",
        "cfo_pos_5y_ratio", "fcf_pos_5y_ratio", "fcf_payout_3y",
        "value_pass", "eps_fwd", "data_quality_flags", "error_flags",
        "rs_percentile", "tt_all_pass", "composite_score",
    ]
    debug_cols_present = [c for c in debug_cols if c in df.columns]
    h_debug = df[df["in_holdings"]][debug_cols_present].copy()

    h_merge = h_df[
        ["code_jquants_5digit", "name_pdf", "account", "shares", "avg_cost_yen"]
    ].merge(h_debug, left_on="code_jquants_5digit", right_on="code", how="left")

    out_h = OUT_DIR / "holdings_debug_fixed_v4.csv"
    h_merge.to_csv(out_h, index=False, encoding="utf-8-sig")
    logger.info("holdings_debug_fixed_v4.csv → %s (%d rows)", out_h, len(h_merge))

    # ── Top lists ─────────────────────────────────────────────────────────────
    score_col = "composite_score" if "composite_score" in df.columns else "rs_percentile"

    core_top = df[df["core_pass"]].sort_values(score_col, ascending=False).head(30)
    core_top.to_csv(OUT_DIR / "core_pass_top30_v4.csv", index=False, encoding="utf-8-sig")
    logger.info("core_pass_top30_v4.csv: %d rows", len(core_top))

    fin_top = df[df["core_fin_pass"]].sort_values(score_col, ascending=False).head(30)
    fin_top.to_csv(OUT_DIR / "core_fin_pass_top30_v4.csv", index=False, encoding="utf-8-sig")
    logger.info("core_fin_pass_top30_v4.csv: %d rows", len(fin_top))

    if "satellite_pass" in df.columns:
        sat_top = df[df["satellite_pass"]].sort_values(score_col, ascending=False).head(20)
        sat_top.to_csv(OUT_DIR / "satellite_pass_top20_v4.csv", index=False, encoding="utf-8-sig")
        logger.info("satellite_pass_top20_v4.csv: %d rows", len(sat_top))

    # ── Final round-trip assertion ────────────────────────────────────────────
    df2 = pd.read_csv(out_v4, encoding="utf-8-sig", low_memory=False)
    assert int(df2["in_holdings"].sum()) == 21, "FAIL: in_holdings round-trip"
    assert int(df2["drop_reason"].isna().sum()) == 0, "FAIL: drop_reason NaN round-trip"

    yield_nan_rt  = df2["dividend_yield_fwd_total"].isna()
    cp_bug_rt     = int((yield_nan_rt & (df2["core_pass"] == True)).sum())
    cfp_bug_rt    = int((yield_nan_rt & (df2["core_fin_pass"] == True)).sum())
    assert cp_bug_rt  == 0, f"FAIL round-trip: yield_total NaN & core_pass=True: {cp_bug_rt}"
    assert cfp_bug_rt == 0, f"FAIL round-trip: yield_total NaN & core_fin_pass=True: {cfp_bug_rt}"

    logger.info("=== All assertions passed ===")
    logger.info(
        "Summary: mode=%s  core_pass=%d  core_fin_pass=%d  satellite_pass=%d  total=%d",
        mode_str,
        int(df2["core_pass"].sum()),
        int(df2["core_fin_pass"].sum()),
        int(df2.get("satellite_pass", pd.Series(dtype=bool)).sum()) if "satellite_pass" in df2 else 0,
        len(df2),
    )


if __name__ == "__main__":
    main()
