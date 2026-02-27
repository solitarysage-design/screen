"""diskcache-based TTL cache with a decorator helper."""
from __future__ import annotations

import json
import functools
import logging
from typing import Any, Callable

import diskcache

from screen.config import CACHE_DIR

logger = logging.getLogger(__name__)

_cache: diskcache.Cache | None = None


def get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(str(CACHE_DIR))
    return _cache


def cached(key_fn: Callable[..., str], ttl: int):
    """Decorator that caches the return value of a function.

    Args:
        key_fn: Callable that receives the same args/kwargs and returns a cache key string.
        ttl: Time-to-live in seconds.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            key = key_fn(*args, **kwargs)
            if key in cache:
                logger.debug("Cache hit: %s", key)
                return cache[key]
            result = func(*args, **kwargs)
            cache.set(key, result, expire=ttl)
            logger.debug("Cache set: %s (ttl=%ds)", key, ttl)
            return result
        return wrapper
    return decorator


def make_key(namespace: str, **params) -> str:
    """Build a deterministic cache key from a namespace and keyword params."""
    serialized = json.dumps(params, sort_keys=True, default=str)
    return f"{namespace}::{serialized}"


def clear_cache() -> None:
    """Evict all entries from the cache."""
    get_cache().clear()
    logger.info("Cache cleared.")
