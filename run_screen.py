"""ワンプッシュ スクリーニング実行ラッパー

Usage:
    python run_screen.py --asof 2026-02-26
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_screen")

# ── Defaults ──────────────────────────────────────────────────────────────────
import os as _os
PROJECT_DIR  = Path(_os.environ.get("SCREEN_PROJECT_DIR", str(Path(__file__).resolve().parent)))
OUTPUT_DIR   = Path(_os.environ.get("SCREEN_OUTPUT_DIR", str(PROJECT_DIR / "output")))
UNIVERSE     = "prime_standard"
HOLDINGS_DIR = Path(_os.environ.get("SCREEN_HOLDINGS_DIR", str(Path.home() / "Downloads")))


def _find_latest_holdings() -> Path:
    """~/Downloads/holdings_extracted_*.csv の最新ファイルを返す。"""
    candidates = sorted(HOLDINGS_DIR.glob("holdings_extracted_*.csv"))
    if not candidates:
        logger.error("holdings_extracted_*.csv が %s に見つかりません", HOLDINGS_DIR)
        sys.exit(1)
    latest = candidates[-1]  # ファイル名の日付ソートで最新
    return latest


def _parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    logger.error("日付パースエラー: %s (YYYY-MM-DD or YYYYMMDD)", s)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ワンプッシュ スクリーニング")
    parser.add_argument("--asof", required=True, help="スクリーニング日 (YYYY-MM-DD)")
    args = parser.parse_args()

    asof_date = _parse_date(args.asof)
    asof_str = asof_date.strftime("%Y-%m-%d")
    dated_dir = OUTPUT_DIR / asof_date.strftime("%Y%m%d")
    holdings_csv = _find_latest_holdings()

    logger.info("=" * 60)
    logger.info("スクリーニング開始")
    logger.info("  asof:     %s", asof_str)
    logger.info("  universe: %s", UNIVERSE)
    logger.info("  holdings: %s", holdings_csv)
    logger.info("  output:   %s", dated_dir)
    logger.info("=" * 60)

    t0 = time.time()

    # ── Step 1: screen CLI ────────────────────────────────────────────────────
    logger.info("[1/3] screen CLI 実行中...")
    cmd = [
        sys.executable, "-m", "screen.cli",
        "--asof", asof_str,
        "--universe", UNIVERSE,
        "--holdings_csv", str(holdings_csv),
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        logger.error("[1/3] screen CLI 失敗 (exit code %d)", result.returncode)
        sys.exit(1)
    logger.info("[1/3] screen CLI 完了")

    # ── Step 2: patch_v2_1 ───────────────────────────────────────────────────
    logger.info("[2/3] patch_v2_1 実行中...")
    import patch_v2_1
    patch_v2_1.INPUT_CSV    = dated_dir / "candidates_fixed_v2.csv"
    patch_v2_1.HOLDINGS_CSV = holdings_csv
    patch_v2_1.OUT_DIR      = dated_dir
    patch_v2_1.main()
    logger.info("[2/3] patch_v2_1 完了")

    # ── Step 3: patch_v2_2 ───────────────────────────────────────────────────
    logger.info("[3/3] patch_v2_2 実行中...")
    import patch_v2_2
    patch_v2_2.INPUT_CSV    = dated_dir / "candidates_fixed_v2_1.csv"
    patch_v2_2.HOLDINGS_CSV = holdings_csv
    patch_v2_2.OUT_DIR      = dated_dir
    patch_v2_2.main()
    logger.info("[3/3] patch_v2_2 完了")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("全ステップ完了 (%.1f秒)", elapsed)

    final_files = [
        dated_dir / "candidates_fixed_v2_2.csv",
        dated_dir / "holdings_debug_v2_2.csv",
        dated_dir / "manual_dividend_check_queue_v2_2.csv",
    ]
    for f in final_files:
        if f.exists():
            size_kb = f.stat().st_size / 1024
            logger.info("  %s (%.0f KB)", f.name, size_kb)
        else:
            logger.warning("  %s — 未生成", f.name)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
