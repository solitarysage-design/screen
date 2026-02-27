"""Universe (listed stock) retrieval."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from screen.data.jquants_client import fetch_listed_info

logger = logging.getLogger(__name__)

SEGMENT_MAP: dict[str, list[str]] = {
    "prime": ["プライム"],
    "standard": ["スタンダード"],
    "prime_standard": ["プライム", "スタンダード"],
    "growth": ["グロース"],
    "all": [],  # no filter
}


def get_universe(
    segment: str = "all",
    custom_csv: str | None = None,
) -> pd.DataFrame:
    """Return universe DataFrame with columns: Code, Name, MarketSegment,
    Sector17CodeName, Sector33CodeName.

    Args:
        segment: Market segment filter. One of "prime", "standard", "growth", "all".
        custom_csv: Path to CSV with a 'Code' column to restrict universe.
    """
    logger.info("Fetching listed info from J-Quants...")
    raw = fetch_listed_info()

    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        df = pd.DataFrame(raw)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Keep only currently listed stocks (filter out delisted)
    if "MarketCodeName" in df.columns:
        df = df[df["MarketCodeName"].notna()]

    # Code を文字列化（J-Quants の株式コードはもともと4桁、切り詰め不要）
    if "Code" in df.columns:
        df["Code"] = df["Code"].astype(str)

    # Segment filter
    segments = SEGMENT_MAP.get(segment, [])
    if segments and "MarketCodeName" in df.columns:
        mask = df["MarketCodeName"].isin(segments)
        before = len(df)
        df = df[mask]
        logger.info("Segment filter '%s': %d → %d stocks", segment, before, len(df))
    elif segments and "MarketSegment" in df.columns:
        mask = df["MarketSegment"].isin(segments)
        before = len(df)
        df = df[mask]
        logger.info("Segment filter '%s': %d → %d stocks", segment, before, len(df))

    # Custom CSV filter
    if custom_csv:
        csv_path = Path(custom_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"custom_csv not found: {custom_csv}")
        custom_df = pd.read_csv(csv_path, dtype=str)
        if "Code" not in custom_df.columns:
            raise ValueError("custom_csv must have a 'Code' column")
        custom_codes = set(custom_df["Code"].str.zfill(4).str[:4])
        before = len(df)
        df = df[df["Code"].isin(custom_codes)]
        logger.info("Custom CSV filter: %d → %d stocks", before, len(df))

    # Select and rename columns to canonical names
    col_renames: dict[str, str] = {}
    for src, dst in [
        ("MarketCodeName", "MarketSegment"),
        ("Sector17CodeName", "Sector17CodeName"),
        ("Sector33CodeName", "Sector33CodeName"),
        ("CompanyName", "Name"),
        ("Name", "Name"),
    ]:
        if src in df.columns and src != dst:
            col_renames[src] = dst

    df = df.rename(columns=col_renames)

    keep_cols = [c for c in ["Code", "Name", "MarketSegment", "Sector17CodeName", "Sector33CodeName"] if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["Code"]).reset_index(drop=True)

    logger.info("Universe: %d stocks", len(df))
    return df
