"""J-Quants API client wrapper with retry and rate limiting."""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, timedelta
from typing import Iterator

import jquantsapi as jq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from screen.config import JQ_CHUNK_DAYS, RATE_LIMIT_RPS

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


_rate_limiter = RateLimiter(RATE_LIMIT_RPS)
_client: jq.Client | None = None


def get_client() -> jq.Client:
    global _client
    if _client is None:
        import os
        # 独自環境変数名 → jquantsapi が期待する引数に橋渡し
        mail = (
            os.environ.get("JQUANTS_API_MAIL_ADDRESS")
            or os.environ.get("JQUANTS_MAIL_ADDRESS", "")
        )
        password = (
            os.environ.get("JQUANTS_API_PASSWORD")
            or os.environ.get("JQUANTS_PASSWORD", "")
        )
        refresh_token = (
            os.environ.get("JQUANTS_API_REFRESH_TOKEN")
            or os.environ.get("JQUANTS_REFRESH_TOKEN", "")
        )
        if refresh_token:
            _client = jq.Client(refresh_token=refresh_token)
        else:
            _client = jq.Client(mail_address=mail, password=password)
    return _client


def _retry_decorator():
    return retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )


def split_date_range(
    start: date, end: date, chunk_days: int = JQ_CHUNK_DAYS
) -> Iterator[tuple[date, date]]:
    """Yield (chunk_start, chunk_end) pairs covering [start, end]."""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


@_retry_decorator()
def fetch_listed_info() -> object:
    _rate_limiter.acquire()
    client = get_client()
    return client.get_listed_info()


@_retry_decorator()
def fetch_daily_quotes(code: str, start: date, end: date) -> object:
    """Fetch daily quotes for a single code over a date range."""
    _rate_limiter.acquire()
    client = get_client()
    return client.get_prices_daily_quotes(
        code=code,
        from_yyyymmdd=start.strftime("%Y%m%d"),
        to_yyyymmdd=end.strftime("%Y%m%d"),
    )


@_retry_decorator()
def fetch_statements(code: str) -> object:
    """Fetch financial statements for a single code."""
    _rate_limiter.acquire()
    client = get_client()
    return client.get_fins_statements(code=code)
