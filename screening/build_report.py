"""Generate a self-contained HTML report from screening outputs.

The report includes:
  - KPI dashboard cards
  - Core verified top-30 table
  - Delta report (new / removed / changed vs previous week)
  - Manual queue summaries
  - Base64-encoded CSV download buttons (for GPT / local analysis)
"""
from __future__ import annotations

import base64
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns to display in the core verified table
_CORE_DISPLAY_COLS = [
    ("code", "Code"),
    ("Name", "Name"),
    ("Sector33CodeName", "Sector"),
    ("effective_yield", "Yield"),
    ("non_cut_years", "NonCut"),
    ("rs_percentile", "RS%"),
    ("trend_score", "Trend"),
    ("core_pass_verified", "Core"),
    ("core_buyable_now_verified", "Buyable"),
    ("in_holdings", "Holdings"),
]

# CSV files to offer as downloads
_DOWNLOAD_FILES = [
    ("candidates_fixed_v2_6.csv", "Full Candidates (GPT用)"),
    ("core_verified_top30_v2_6.csv", "Core Verified Top30"),
    ("manual_shares_check_queue_v2_6.csv", "Manual Shares Queue"),
    ("manual_dividend_check_queue_v2_6.csv", "Manual Dividend Queue"),
    ("data_fill_queue_v2_6.csv", "Data Fill Queue"),
    ("weekly_delta_report_v2_6.csv", "Delta Report"),
    ("holdings_debug_v2_6.csv", "Holdings Debug"),
]


def _b64_csv(path: Path) -> str:
    """Read a CSV file and return base64-encoded content for data URI."""
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _fmt_pct(v) -> str:
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return str(v) if pd.notna(v) else "-"


def _fmt_num(v) -> str:
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return str(v) if pd.notna(v) else "-"


def _fmt_bool(v) -> str:
    s = str(v).lower()
    if s in ("true", "1", "1.0"):
        return "Yes"
    if s in ("false", "0", "0.0"):
        return ""
    return str(v) if pd.notna(v) else "-"


def _build_kpi_cards(kpi: dict) -> str:
    cards = []

    def _card(label: str, value, color: str = "#2563eb") -> str:
        return (
            f'<div class="kpi-card">'
            f'<div class="kpi-value" style="color:{color}">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>'
        )

    cards.append(_card("Total Candidates", kpi.get("total_candidates", "?")))
    cards.append(_card("Core Verified", kpi.get("core_pass_verified", "?"), "#16a34a"))
    cards.append(_card("Shares Queue", kpi.get("manual_shares_check", "?"), "#d97706"))
    cards.append(_card("Dividend Queue", kpi.get("manual_dividend_check", "?"), "#d97706"))
    cards.append(_card("Data Fill", kpi.get("data_fill", "?"), "#dc2626"))

    if "delta_new" in kpi:
        cards.append(_card("New", f"+{kpi['delta_new']}", "#16a34a"))
        cards.append(_card("Removed", f"-{kpi['delta_removed']}", "#dc2626"))
        cards.append(_card("Changed", kpi["delta_changed"], "#d97706"))

    elapsed = kpi.get("elapsed_seconds")
    if elapsed:
        cards.append(_card("Elapsed", f"{elapsed / 60:.0f}min", "#6b7280"))

    return "\n".join(cards)


def _build_table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    """Build an HTML table from a DataFrame with specified columns."""
    existing = [(col, label) for col, label in columns if col in df.columns]
    if not existing:
        return "<p>No data</p>"

    rows_html = []
    # Header
    header = "".join(f"<th>{label}</th>" for _, label in existing)
    rows_html.append(f"<tr>{header}</tr>")

    # Body
    for _, row in df.iterrows():
        cells = []
        for col, _ in existing:
            v = row[col]
            if col == "effective_yield":
                cells.append(f"<td>{_fmt_pct(v)}</td>")
            elif col in ("rs_percentile", "trend_score", "non_cut_years"):
                cells.append(f"<td>{_fmt_num(v)}</td>")
            elif col in ("core_pass_verified", "core_buyable_now_verified",
                         "satellite_buyable_now_verified", "in_holdings"):
                cls = "yes" if _fmt_bool(v) == "Yes" else ""
                cells.append(f'<td class="{cls}">{_fmt_bool(v)}</td>')
            else:
                cells.append(f"<td>{v if pd.notna(v) else '-'}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return f'<table>{"".join(rows_html)}</table>'


def _build_delta_table(df: pd.DataFrame) -> str:
    """Build delta report table."""
    if df is None or df.empty:
        return "<p>Delta data not available (first run or no changes)</p>"

    rows_html = []
    rows_html.append("<tr><th>Type</th><th>Code</th><th>Name</th><th>Detail</th></tr>")

    type_colors = {"new": "#16a34a", "removed": "#dc2626", "changed": "#d97706"}
    type_labels = {"new": "NEW", "removed": "REMOVED", "changed": "CHANGED"}

    for _, row in df.iterrows():
        dtype = str(row.get("delta_type", ""))
        color = type_colors.get(dtype, "#6b7280")
        label = type_labels.get(dtype, dtype)
        code = row.get("code", "")
        name = row.get("Name", "")
        detail = row.get("detail", "")
        rows_html.append(
            f'<tr>'
            f'<td><span class="badge" style="background:{color}">{label}</span></td>'
            f'<td>{code}</td><td>{name}</td><td>{detail}</td>'
            f'</tr>'
        )

    return f'<table>{"".join(rows_html)}</table>'


def _build_download_buttons(output_dir: Path) -> str:
    """Build CSV download buttons with base64 data URIs."""
    buttons = []
    for filename, label in _DOWNLOAD_FILES:
        path = output_dir / filename
        if path.exists():
            b64 = _b64_csv(path)
            buttons.append(
                f'<a class="dl-btn" '
                f'href="data:text/csv;base64,{b64}" '
                f'download="{filename}">'
                f'{label}'
                f'</a>'
            )
    return "\n".join(buttons)


_CSS = """\
:root {
  --bg: #f8fafc; --fg: #0f172a; --card-bg: #fff;
  --border: #e2e8f0; --accent: #2563eb; --muted: #64748b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a; --fg: #e2e8f0; --card-bg: #1e293b;
    --border: #334155; --accent: #60a5fa; --muted: #94a3b8;
  }
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg); color: var(--fg);
  line-height: 1.6; padding: 2rem; max-width: 1200px; margin: 0 auto;
}
h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
h2 { font-size: 1.2rem; margin: 2rem 0 0.75rem; border-bottom: 2px solid var(--border); padding-bottom: 0.3rem; }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; }
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
  gap: 0.75rem; margin-bottom: 1.5rem;
}
.kpi-card {
  background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem; text-align: center;
}
.kpi-value { font-size: 1.8rem; font-weight: 700; }
.kpi-label { font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }
table {
  width: 100%; border-collapse: collapse; background: var(--card-bg);
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
  font-size: 0.85rem; margin-bottom: 1rem;
}
th { background: var(--border); font-weight: 600; text-align: left; padding: 0.5rem 0.75rem; }
td { padding: 0.4rem 0.75rem; border-top: 1px solid var(--border); }
tr:hover td { background: color-mix(in srgb, var(--accent) 8%, transparent); }
td.yes { color: #16a34a; font-weight: 600; }
.badge {
  display: inline-block; color: #fff; padding: 0.15rem 0.5rem;
  border-radius: 4px; font-size: 0.75rem; font-weight: 600;
}
.dl-section { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }
.dl-btn {
  display: inline-block; padding: 0.6rem 1.2rem;
  background: var(--accent); color: #fff; text-decoration: none;
  border-radius: 6px; font-size: 0.85rem; font-weight: 500;
  transition: opacity 0.2s;
}
.dl-btn:hover { opacity: 0.85; }
footer { margin-top: 3rem; color: var(--muted); font-size: 0.8rem; text-align: center; }
"""


def build_html_report(output_dir: Path, kpi: dict) -> Path:
    """Generate a self-contained HTML report and return its path."""
    asof = kpi.get("date", str(date.today()))

    # Load dataframes
    core_top30_path = output_dir / "core_verified_top30_v2_6.csv"
    core_df = pd.read_csv(core_top30_path, dtype=str) if core_top30_path.exists() else pd.DataFrame()

    delta_path = output_dir / "weekly_delta_report_v2_6.csv"
    delta_df = pd.read_csv(delta_path, dtype=str) if delta_path.exists() else None

    candidates_path = output_dir / "candidates_fixed_v2_6.csv"
    cand_df = pd.read_csv(candidates_path, dtype=str) if candidates_path.exists() else pd.DataFrame()

    # Holdings summary from candidates
    holdings_section = ""
    if not cand_df.empty and "in_holdings" in cand_df.columns:
        h_df = cand_df[cand_df["in_holdings"].astype(str).str.lower().isin(["true", "1", "1.0"])]
        if not h_df.empty:
            holdings_cols = [
                ("code", "Code"),
                ("Name", "Name"),
                ("effective_yield", "Yield"),
                ("non_cut_years", "NonCut"),
                ("core_pass_verified", "Core"),
                ("core_buyable_now_verified", "Buyable"),
                ("drop_reason_core", "Drop Reason"),
            ]
            holdings_section = (
                '<h2>Holdings Status</h2>'
                + _build_table(h_df, holdings_cols)
            )

    html = f"""\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weekly Screening Report - {asof}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Weekly Screening Report</h1>
<p class="subtitle">{asof}</p>

<div class="kpi-grid">
{_build_kpi_cards(kpi)}
</div>

<h2>CSV Downloads</h2>
<div class="dl-section">
{_build_download_buttons(output_dir)}
</div>

<h2>Core Verified Top 30</h2>
{_build_table(core_df, _CORE_DISPLAY_COLS)}

{holdings_section}

<h2>Delta vs Previous Week</h2>
{_build_delta_table(delta_df)}

<footer>
  Generated by screening/build_report.py
</footer>
</body>
</html>
"""

    out_path = output_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("HTML report generated: %s", out_path)
    return out_path
