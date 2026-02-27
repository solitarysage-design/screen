"""Multi-source dividend history fetcher.

Priority: J-Quants → IRBank → yfinance.

Unit contract: all DPS values are per-share, split-adjusted, in JPY.
Annual DPS = calendar-year sum of per-share dividends.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Literal

import pandas as pd
import requests

from screen.data.cache import get_cache, make_key
from screen.config import CACHE_TTL_FUNDAMENTALS

logger = logging.getLogger(__name__)

DivSourceType = Literal["jquants", "irbank", "yf"]

# Minimum coverage to consider a source "sufficient"
MIN_COVERAGE_YEARS = 5

# IRBank request timeout
_IRBANK_TIMEOUT = 10
_IRBANK_BASE = "https://irbank.net"

# yfinance TTM window (days)
_TTM_DAYS = 365


@dataclass
class AnnualDivRecord:
    year: int
    dps: float  # per-share, split-adjusted, JPY


@dataclass
class DivHistoryResult:
    code: str
    annual_records: list[AnnualDivRecord] = field(default_factory=list)
    source_used: DivSourceType = "jquants"
    coverage_years: int = 0
    has_splits: bool = False
    split_adjusted: bool = False

    @property
    def dps_list_desc(self) -> list[float]:
        """DPS values sorted most-recent-first."""
        return [r.dps for r in sorted(self.annual_records, key=lambda r: r.year, reverse=True)]

    def ttm_dps(self) -> float | None:
        """Most recent annual DPS (proxy for TTM)."""
        recs = sorted(self.annual_records, key=lambda r: r.year, reverse=True)
        return recs[0].dps if recs else None


# ── J-Quants source ───────────────────────────────────────────────────────────

def _from_jquants(
    code: str,
    jq_dps_list: list[float | None] | None,
    jq_fy_dates: list[str] | None,
) -> DivHistoryResult | None:
    """Build DivHistoryResult from J-Quants FY statements.

    jq_dps_list: ResultDividendPerShareAnnual, most-recent-first.
    jq_fy_dates: CurrentFiscalYearEndDate strings (YYYY-MM-DD), most-recent-first.
    """
    if not jq_dps_list:
        return None

    records: list[AnnualDivRecord] = []
    for i, dps in enumerate(jq_dps_list):
        if dps is None:
            continue
        # Derive calendar year from FY end date
        year: int | None = None
        if jq_fy_dates and i < len(jq_fy_dates):
            try:
                year = pd.to_datetime(jq_fy_dates[i]).year
            except Exception:
                pass
        if year is None:
            # Fallback: use index offset from current year
            import datetime
            year = datetime.date.today().year - i
        records.append(AnnualDivRecord(year=year, dps=float(dps)))

    if not records:
        return None

    # Deduplicate by year (keep first = most-recent FY end for that year)
    seen: set[int] = set()
    deduped: list[AnnualDivRecord] = []
    for r in records:
        if r.year not in seen:
            seen.add(r.year)
            deduped.append(r)

    return DivHistoryResult(
        code=code,
        annual_records=sorted(deduped, key=lambda r: r.year, reverse=True),
        source_used="jquants",
        coverage_years=len(deduped),
        has_splits=False,
        split_adjusted=False,
    )


# ── IRBank source ─────────────────────────────────────────────────────────────

class _TableParser(HTMLParser):
    """Minimal HTML table extractor."""

    def __init__(self) -> None:
        super().__init__()
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_row: list[str] = []
        self._current_text: str = ""
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "table":
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._current_row = []
        elif tag in ("td", "th") and self._in_row:
            self._in_cell = True
            self._current_text = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._in_table = False
        elif tag == "tr" and self._in_row:
            self._in_row = False
            if self._current_row:
                self.rows.append(self._current_row[:])
        elif tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            self._current_row.append(self._current_text.strip())

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_text += data


def _parse_irbank_divs(html: str) -> list[tuple[int, float]]:
    """Extract (year, annual_dps) pairs from IRBank dividend HTML.

    IRBank table typically has columns: 年度, 中間, 期末, 合計(or 1株配当), ...
    We look for rows where column 0 contains a 4-digit year and
    the last numeric column is the annual DPS total.
    """
    parser = _TableParser()
    parser.feed(html)

    results: list[tuple[int, float]] = []

    for row in parser.rows:
        if len(row) < 2:
            continue
        # Check if first cell contains a year
        year_match = re.search(r"(\d{4})", row[0])
        if not year_match:
            continue
        year = int(year_match.group(1))
        if year < 2000 or year > 2040:
            continue

        # Find the rightmost numeric-looking cell (= annual total DPS)
        dps: float | None = None
        for cell in reversed(row[1:]):
            clean = re.sub(r"[,\s円]", "", cell)
            # Skip cells that look like a percentage or non-DPS
            if re.match(r"^-?\d+(\.\d+)?$", clean):
                try:
                    val = float(clean)
                    if val >= 0:
                        dps = val
                        break
                except ValueError:
                    pass

        if dps is not None:
            results.append((year, dps))

    return results


def _fetch_irbank_raw(code4: str) -> str | None:
    """Fetch raw HTML from IRBank dividends page. Cached."""
    cache = get_cache()
    key = make_key("irbank_divs", code=code4)
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    url = f"{_IRBANK_BASE}/{code4}/divs"
    try:
        resp = requests.get(
            url,
            timeout=_IRBANK_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (screen-app/1.0)"},
        )
        if resp.status_code != 200:
            logger.debug("IRBank %s: HTTP %d", code4, resp.status_code)
            return None
        html = resp.text
        cache.set(key, html, expire=CACHE_TTL_FUNDAMENTALS)
        return html
    except Exception as exc:
        logger.debug("IRBank fetch failed for %s: %s", code4, exc)
        return None


def _from_irbank(code: str) -> DivHistoryResult | None:
    """Fetch annual DPS history from IRBank."""
    # J-Quants codes are 5 digits (4-digit + trailing "0")
    code4 = code[:-1] if len(code) == 5 and code.endswith("0") else code

    html = _fetch_irbank_raw(code4)
    if not html:
        return None

    pairs = _parse_irbank_divs(html)
    if not pairs:
        logger.debug("IRBank %s: no dividend records parsed", code4)
        return None

    records = [AnnualDivRecord(year=yr, dps=dps) for yr, dps in pairs]
    records = sorted(records, key=lambda r: r.year, reverse=True)

    # Deduplicate by year
    seen: set[int] = set()
    deduped = []
    for r in records:
        if r.year not in seen:
            seen.add(r.year)
            deduped.append(r)

    logger.debug("IRBank %s: %d annual records", code4, len(deduped))
    return DivHistoryResult(
        code=code,
        annual_records=deduped,
        source_used="irbank",
        coverage_years=len(deduped),
        has_splits=False,
        split_adjusted=False,
    )


# ── yfinance source ───────────────────────────────────────────────────────────

def _from_yfinance(code: str) -> DivHistoryResult | None:
    """Fetch annual DPS history from yfinance with split adjustment.

    Uses history(auto_adjust=False) to get raw (unadjusted) dividends,
    then applies cumulative forward split factor to normalize to current-share basis.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    code4 = code[:-1] if len(code) == 5 and code.endswith("0") else code
    ticker_sym = f"{code4}.T"

    try:
        ticker = yf.Ticker(ticker_sym)

        # Raw (unadjusted) dividend history
        hist = ticker.history(period="max", auto_adjust=False, actions=True)
        if hist.empty or "Dividends" not in hist.columns:
            return None

        raw_divs = hist["Dividends"].dropna()
        raw_divs = raw_divs[raw_divs > 0]
        if raw_divs.empty:
            return None

        # Stock splits
        splits_series = hist["Stock Splits"].dropna() if "Stock Splits" in hist.columns else pd.Series(dtype=float)
        splits_series = splits_series[splits_series > 0]
        has_splits = not splits_series.empty

        # Compute cumulative forward split factor for each dividend date
        # adj_dps = raw_dps * product(split_ratios after that date)
        if has_splits:
            adj_divs_dict: dict = {}
            for div_date, raw_dps in raw_divs.items():
                future_splits = splits_series[splits_series.index > div_date]
                factor = float(future_splits.prod()) if not future_splits.empty else 1.0
                adj_divs_dict[div_date] = raw_dps * factor
            adj_divs = pd.Series(adj_divs_dict, dtype=float)
            split_adjusted = True
        else:
            adj_divs = raw_divs.copy()
            split_adjusted = False

        # Resample to calendar year
        try:
            annual = adj_divs.resample("YE").sum()
        except Exception:
            annual = adj_divs.resample("A").sum()

        annual = annual[annual > 0]
        if annual.empty:
            return None

        records = [
            AnnualDivRecord(year=int(dt.year), dps=float(dps))
            for dt, dps in annual.items()
        ]
        records = sorted(records, key=lambda r: r.year, reverse=True)

        logger.debug("yfinance %s: %d annual records (splits=%s)", code4, len(records), has_splits)
        return DivHistoryResult(
            code=code,
            annual_records=records,
            source_used="yf",
            coverage_years=len(records),
            has_splits=has_splits,
            split_adjusted=split_adjusted,
        )

    except Exception as exc:
        logger.debug("yfinance div history failed for %s: %s", code4, exc)
        return None


# ── TTM yield helper (yfinance) ───────────────────────────────────────────────

def get_ttm_dps_yf(code: str) -> float | None:
    """Compute TTM DPS from yfinance actions/dividends (split-adjusted).

    Uses last 365 days of dividend records to avoid trailingAnnualDividendRate
    mismatch issues.
    """
    try:
        import yfinance as yf
        import datetime

        code4 = code[:-1] if len(code) == 5 and code.endswith("0") else code
        ticker = yf.Ticker(f"{code4}.T")

        hist = ticker.history(period="2y", auto_adjust=False, actions=True)
        if hist.empty or "Dividends" not in hist.columns:
            return None

        raw_divs = hist["Dividends"].dropna()
        raw_divs = raw_divs[raw_divs > 0]
        if raw_divs.empty:
            return None

        splits_series = hist["Stock Splits"].dropna() if "Stock Splits" in hist.columns else pd.Series(dtype=float)
        splits_series = splits_series[splits_series > 0]

        # Adjust for splits
        if not splits_series.empty:
            adj_dict: dict = {}
            for div_date, raw_dps in raw_divs.items():
                future_splits = splits_series[splits_series.index > div_date]
                factor = float(future_splits.prod()) if not future_splits.empty else 1.0
                adj_dict[div_date] = raw_dps * factor
            adj_divs = pd.Series(adj_dict, dtype=float)
        else:
            adj_divs = raw_divs.copy()

        # TTM: sum of last 365 days
        cutoff = adj_divs.index.max() - pd.Timedelta(days=_TTM_DAYS)
        ttm_divs = adj_divs[adj_divs.index >= cutoff]
        return float(ttm_divs.sum()) if not ttm_divs.empty else None

    except Exception as exc:
        logger.debug("TTM DPS yfinance failed for %s: %s", code, exc)
        return None


# ── Main orchestrator ─────────────────────────────────────────────────────────

def get_div_history(
    code: str,
    jq_dps_list: list[float | None] | None = None,
    jq_fy_dates: list[str] | None = None,
) -> DivHistoryResult:
    """Fetch annual dividend history using best available source.

    Priority: J-Quants → IRBank → yfinance.
    Falls back to lower-priority sources when coverage_years < MIN_COVERAGE_YEARS.

    Returns DivHistoryResult with the source that has the most coverage,
    preferring higher-priority sources when coverage is equal.
    """
    best: DivHistoryResult | None = None

    # ── Source 1: J-Quants ────────────────────────────────────────────────
    jq_result = _from_jquants(code, jq_dps_list, jq_fy_dates)
    if jq_result:
        best = jq_result
        if best.coverage_years >= MIN_COVERAGE_YEARS:
            logger.debug("%s: J-Quants sufficient (coverage=%d)", code, best.coverage_years)
            return best

    # ── Source 2: IRBank ──────────────────────────────────────────────────
    try:
        irbank_result = _from_irbank(code)
        if irbank_result:
            if best is None or irbank_result.coverage_years > best.coverage_years:
                best = irbank_result
                logger.debug("%s: IRBank better (coverage=%d)", code, best.coverage_years)
            if best.coverage_years >= MIN_COVERAGE_YEARS:
                return best
    except Exception as exc:
        logger.debug("%s: IRBank error: %s", code, exc)

    # ── Source 3: yfinance ────────────────────────────────────────────────
    try:
        yf_result = _from_yfinance(code)
        if yf_result:
            if best is None or yf_result.coverage_years > best.coverage_years:
                best = yf_result
                logger.debug("%s: yfinance better (coverage=%d)", code, best.coverage_years)
    except Exception as exc:
        logger.debug("%s: yfinance error: %s", code, exc)

    if best is None:
        logger.debug("%s: no dividend history from any source", code)
        return DivHistoryResult(code=code, coverage_years=0, source_used="jquants")

    logger.debug(
        "%s: best source=%s coverage=%d",
        code, best.source_used, best.coverage_years,
    )
    return best
