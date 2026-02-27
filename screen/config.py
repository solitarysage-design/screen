"""Global constants, flags, and thresholds."""
from __future__ import annotations

import os
from pathlib import Path

# --- SMA periods ---
SMA_PERIODS: list[int] = [50, 150, 200]

# --- Relative Strength ---
RS_WEIGHTS: dict[str, float] = {"3m": 0.4, "6m": 0.3, "9m": 0.2, "12m": 0.1}
RS_MIN_PERCENTILE: float = 70.0

# --- Minervini Trend Template thresholds ---
TT_PRICE_VS_LOW52W: float = 1.30   # price >= low52w * 1.30
TT_PRICE_VS_HIGH52W: float = 0.75  # price >= high52w * 0.75
TT_SMA200_LOOKBACK: int = 20       # days to check SMA200 trend

# --- FCF Payout ---
FCF_PAYOUT_HARD: float = 0.70       # non-cyclical hard cap
FCF_PAYOUT_CYCLICAL: float = 0.60   # cyclical hard cap

# --- Unknown data policy ---
UNKNOWN_POLICY: str = "exclude"  # "include" | "exclude"

# --- Rate limiting ---
RATE_LIMIT_RPS: float = 2.0  # J-Quants requests per second

# --- Cache TTLs (seconds) ---
CACHE_TTL_PRICES: int = 6 * 3600       # 6 hours
CACHE_TTL_FUNDAMENTALS: int = 24 * 3600  # 24 hours

# --- Cache directory ---
CACHE_DIR: Path = Path.home() / ".cache" / "screen"

# --- Output directory ---
OUTPUT_DIR: Path = Path(os.environ.get("SCREEN_OUTPUT_DIR", "C:/Users/solit/projects/screen/output"))

# --- Data lookback ---
DEFAULT_LOOKBACK_DAYS: int = 420

# --- Breakout parameters ---
BREAKOUT_VOLUME_MULT: float = 1.5
BREAKOUT_SHORT_DAYS: int = 20
BREAKOUT_LONG_DAYS: int = 55
PULLBACK_WINDOW_DAYS: int = 40
PULLBACK_SMA50_BAND: float = 0.03  # ±3%
PULLBACK_REBOUND_DAYS: int = 3

# --- Data quality ---
MAX_MISSING_PRICE_RATIO: float = 0.20  # exclude if >20% missing

# --- J-Quants date chunk ---
JQ_CHUNK_DAYS: int = 30

# --- Cyclical sector codes (Sector33) ---
CYCLICAL_SECTOR33: set[str] = {
    "鉄鋼",
    "非鉄金属",
    "石油・石炭製品",
    "化学",
    "海運業",
    "空運業",
    "鉱業",
    "建設業",
}

# --- Financial sector codes (Sector33) ---
# CFO/FCF metrics are structurally inapplicable to banks/insurance.
# These stocks use core_fin_pass instead of core_pass.
FINANCIAL_SECTOR33: set[str] = {
    "銀行業",
    "保険業",
    "その他金融業",
    "証券、商品先物取引業",  # J-Quants 33-sector label (full name with 、)
}
