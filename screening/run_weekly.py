"""Weekly screening automation entry point.

Usage:
    python -m screening.run_weekly                  # today's date
    python -m screening.run_weekly --asof 2026-03-03
    SCREEN_ASOF_DATE=2026-03-03 python -m screening.run_weekly
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_weekly")

# ── Project root (this file lives in  <project>/screening/run_weekly.py) ──
PROJECT_DIR = Path(__file__).resolve().parent.parent

# ── Default holdings CSV ──
DEFAULT_HOLDINGS = PROJECT_DIR / "data" / "holdings.csv"

# ── Top-level outputs directory ──
OUTPUTS_ROOT = PROJECT_DIR / "outputs"

# ── Universe ──
UNIVERSE = "prime_standard"

# ── v2_5 → v2_6 rename map ──
V25_TO_V26_RENAME: dict[str, str] = {
    "candidates_fixed_v2_5.csv": "candidates_fixed_v2_6.csv",
    "core_verified_top30_v2_5.csv": "core_verified_top30_v2_6.csv",
    "manual_shares_check_queue_v2_5.csv": "manual_shares_check_queue_v2_6.csv",
    "manual_dividend_check_queue_v2_5.csv": "manual_dividend_check_queue_v2_6.csv",
    "data_fill_queue_v2_5.csv": "data_fill_queue_v2_6.csv",
    "holdings_debug_v2_5.csv": "holdings_debug_v2_6.csv",
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def _parse_date(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s}")


def _resolve_asof() -> date:
    """Determine the screening date from args / env / today."""
    # 1) CLI argument
    import argparse
    parser = argparse.ArgumentParser(description="Weekly screening runner")
    parser.add_argument("--asof", default=None, help="Screening date (YYYY-MM-DD)")
    args, _ = parser.parse_known_args()
    if args.asof:
        return _parse_date(args.asof)
    # 2) Environment variable
    env_date = os.environ.get("SCREEN_ASOF_DATE", "").strip()
    if env_date:
        return _parse_date(env_date)
    # 3) Today
    return date.today()


def _run(cmd: list[str], label: str, env: dict[str, str] | None = None) -> None:
    """Run a subprocess, raise on failure."""
    merged_env = {**os.environ, **(env or {})}
    logger.info("[%s] %s", label, " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=merged_env)
    if result.returncode != 0:
        raise RuntimeError(f"[{label}] failed with exit code {result.returncode}")
    logger.info("[%s] done", label)


def _rename_v25_to_v26(output_dir: Path) -> None:
    """Rename v2_5 output files to v2_6."""
    for old_name, new_name in V25_TO_V26_RENAME.items():
        src = output_dir / old_name
        dst = output_dir / new_name
        if src.exists():
            shutil.copy2(str(src), str(dst))
            logger.info("Renamed: %s -> %s", old_name, new_name)
        else:
            logger.warning("v2_5 file not found for rename: %s", old_name)


def _compute_delta(output_dir: Path, prev_dir: Path) -> Path | None:
    """Compare current and previous candidates to produce a delta report."""
    current_file = output_dir / "candidates_fixed_v2_6.csv"
    prev_file = prev_dir / "candidates_fixed_v2_6.csv"

    if not prev_file.exists():
        logger.info("No previous candidates found at %s — skipping delta", prev_file)
        return None

    if not current_file.exists():
        logger.warning("Current candidates not found: %s", current_file)
        return None

    df_cur = pd.read_csv(current_file, dtype={"code": str})
    df_prev = pd.read_csv(prev_file, dtype={"code": str})

    cur_codes = set(df_cur["code"].dropna())
    prev_codes = set(df_prev["code"].dropna())

    new_codes = cur_codes - prev_codes
    removed_codes = prev_codes - cur_codes
    common_codes = cur_codes & prev_codes

    rows: list[dict] = []

    for code in sorted(new_codes):
        row = df_cur[df_cur["code"] == code].iloc[0]
        rows.append({
            "code": code,
            "Name": row.get("Name", ""),
            "delta_type": "new",
            "detail": "",
        })

    for code in sorted(removed_codes):
        row = df_prev[df_prev["code"] == code].iloc[0]
        rows.append({
            "code": code,
            "Name": row.get("Name", ""),
            "delta_type": "removed",
            "detail": "",
        })

    # Check for status changes on common codes
    status_cols = [
        "core_pass", "core_pass_verified",
        "core_buyable_now", "core_buyable_now_verified",
        "satellite_buyable_now", "satellite_buyable_now_verified",
    ]
    existing_cols = [c for c in status_cols if c in df_cur.columns and c in df_prev.columns]

    if existing_cols:
        merged = df_cur[df_cur["code"].isin(common_codes)].set_index("code")[existing_cols]
        merged_prev = df_prev[df_prev["code"].isin(common_codes)].set_index("code")[existing_cols]
        for code in sorted(common_codes):
            if code in merged.index and code in merged_prev.index:
                cur_row = merged.loc[code]
                prev_row = merged_prev.loc[code]
                changes = []
                for col in existing_cols:
                    cv = str(cur_row.get(col, ""))
                    pv = str(prev_row.get(col, ""))
                    if cv != pv:
                        changes.append(f"{col}: {pv} -> {cv}")
                if changes:
                    name = df_cur[df_cur["code"] == code].iloc[0].get("Name", "")
                    rows.append({
                        "code": code,
                        "Name": name,
                        "delta_type": "changed",
                        "detail": "; ".join(changes),
                    })

    if not rows:
        logger.info("No delta detected between current and previous runs")
        return None

    delta_df = pd.DataFrame(rows)
    out_path = output_dir / "weekly_delta_report_v2_6.csv"
    delta_df.to_csv(out_path, index=False)
    logger.info("Delta report: %s (%d rows: %d new, %d removed, %d changed)",
                out_path, len(rows),
                len(new_codes), len(removed_codes),
                len(rows) - len(new_codes) - len(removed_codes))
    return out_path


def _copy_to_latest(output_dir: Path) -> Path:
    """Copy all output files to outputs/latest/."""
    latest_dir = OUTPUTS_ROOT / "latest"
    if latest_dir.exists():
        shutil.rmtree(str(latest_dir))
    shutil.copytree(str(output_dir), str(latest_dir))
    logger.info("Copied output to %s", latest_dir)
    return latest_dir


def _build_kpi_summary(output_dir: Path) -> dict:
    """Read key output CSVs and produce KPI summary."""
    kpi: dict = {"date": str(date.today()), "status": "success"}

    candidates = output_dir / "candidates_fixed_v2_6.csv"
    if candidates.exists():
        df = pd.read_csv(candidates, dtype=str)
        kpi["total_candidates"] = len(df)

        for col in ["core_pass_verified", "core_buyable_now_verified",
                     "satellite_buyable_now_verified"]:
            if col in df.columns:
                kpi[col] = int(df[col].astype(str).str.lower().isin(["true", "1"]).sum())

    for queue_name in ["manual_shares_check_queue_v2_6.csv",
                       "manual_dividend_check_queue_v2_6.csv",
                       "data_fill_queue_v2_6.csv"]:
        queue_path = output_dir / queue_name
        key = queue_name.replace("_queue_v2_6.csv", "").replace("_v2_6.csv", "")
        if queue_path.exists():
            qdf = pd.read_csv(queue_path)
            kpi[key] = len(qdf)

    delta_path = output_dir / "weekly_delta_report_v2_6.csv"
    if delta_path.exists():
        ddf = pd.read_csv(delta_path, dtype=str)
        kpi["delta_new"] = int((ddf["delta_type"] == "new").sum())
        kpi["delta_removed"] = int((ddf["delta_type"] == "removed").sum())
        kpi["delta_changed"] = int((ddf["delta_type"] == "changed").sum())

    return kpi


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()

    # 1. Resolve dates and paths
    asof = _resolve_asof()
    asof_str = asof.strftime("%Y-%m-%d")
    date_dir_name = asof.strftime("%Y%m%d")
    output_dir = OUTPUTS_ROOT / asof.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    holdings_csv = Path(os.environ.get("SCREEN_HOLDINGS_CSV", str(DEFAULT_HOLDINGS)))

    logger.info("=" * 60)
    logger.info("Weekly Screening Start")
    logger.info("  asof:     %s", asof_str)
    logger.info("  output:   %s", output_dir)
    logger.info("  holdings: %s", holdings_csv)
    logger.info("  project:  %s", PROJECT_DIR)
    logger.info("=" * 60)

    # Common env vars for subprocesses
    sub_env = {
        "SCREEN_PROJECT_DIR": str(PROJECT_DIR),
        "SCREEN_OUTPUT_DIR": str(output_dir),
        "SCREEN_HOLDINGS_CSV": str(holdings_csv),
    }

    # 2. Step 1: screen CLI
    _run(
        [sys.executable, "-m", "screen.cli",
         "--asof", asof_str,
         "--universe", UNIVERSE,
         "--holdings_csv", str(holdings_csv)],
        label="1/7 screen.cli",
        env=sub_env,
    )

    # 3. Step 2: patch_v2_1
    _run(
        [sys.executable, str(PROJECT_DIR / "patch_v2_1.py")],
        label="2/7 patch_v2_1",
        env=sub_env,
    )

    # 4. Step 3: patch_v2_2
    _run(
        [sys.executable, str(PROJECT_DIR / "patch_v2_2.py")],
        label="3/7 patch_v2_2",
        env=sub_env,
    )

    # 5. Step 4: patch_v2_3
    _run(
        [sys.executable, str(PROJECT_DIR / "patch_v2_3.py")],
        label="4/7 patch_v2_3",
        env=sub_env,
    )

    # 6. Step 5: patch_v2_4
    _run(
        [sys.executable, str(PROJECT_DIR / "patch_v2_4.py"),
         "--input", str(output_dir / "candidates_fixed_v2_3.csv")],
        label="5/7 patch_v2_4",
        env=sub_env,
    )

    # 7. Step 6: patch_v2_5
    _run(
        [sys.executable, str(PROJECT_DIR / "patch_v2_5.py")],
        label="6/7 patch_v2_5",
        env=sub_env,
    )

    # 8. Step 7: Rename v2_5 -> v2_6
    logger.info("[7/7] Renaming v2_5 -> v2_6...")
    _rename_v25_to_v26(output_dir)
    logger.info("[7/7] Rename done")

    # 9. Delta calculation (if prev exists)
    prev_dir = OUTPUTS_ROOT / "prev"
    _compute_delta(output_dir, prev_dir)

    # 10. Copy to outputs/latest/
    _copy_to_latest(output_dir)

    # 11. KPI summary
    kpi = _build_kpi_summary(output_dir)
    elapsed = time.time() - t0
    kpi["elapsed_seconds"] = round(elapsed, 1)

    kpi_path = output_dir / "kpi_summary.json"
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump(kpi, f, ensure_ascii=False, indent=2)
    logger.info("KPI summary saved: %s", kpi_path)

    # Also save to latest/
    latest_kpi = OUTPUTS_ROOT / "latest" / "kpi_summary.json"
    with open(latest_kpi, "w", encoding="utf-8") as f:
        json.dump(kpi, f, ensure_ascii=False, indent=2)

    # 12. Generate HTML report
    from screening.build_report import build_html_report
    report_path = build_html_report(output_dir, kpi)
    # Copy report to latest/
    shutil.copy2(str(report_path), str(OUTPUTS_ROOT / "latest" / "report.html"))

    # Print KPI summary
    logger.info("=" * 60)
    logger.info("Weekly Screening Complete (%.1fs)", elapsed)
    for k, v in kpi.items():
        if k not in ("date", "status", "elapsed_seconds"):
            logger.info("  %s: %s", k, v)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
