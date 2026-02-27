"""HTML レポート生成 - ソート可能テーブル版."""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _row_to_dict(row) -> dict:
    code_raw = str(row.get("code", ""))
    code_disp = (
        code_raw[:-1]
        if len(code_raw) == 5 and code_raw.endswith("0") and code_raw[:4].isdigit()
        else code_raw
    )
    return {
        "code": code_raw,
        "code_disp": code_disp,
        "name": str(row.get("Name") or "—"),
        "sector": str(row.get("Sector33CodeName") or row.get("sector33") or "—"),
        "composite_score": _safe_float(row.get("composite_score")),
        "rs_percentile": _safe_float(row.get("rs_percentile")),
        "dividend_yield_fwd": _safe_float(row.get("dividend_yield_fwd")),
        "per_fwd": _safe_float(row.get("per_fwd")),
        "pbr": _safe_float(row.get("pbr")),
        "non_cut_years": _safe_float(row.get("non_cut_years")),
        "fcf_payout_3y": _safe_float(row.get("fcf_payout_3y")),
        "first_pullback": bool(row.get("first_pullback")),
        "breakout_20d": bool(row.get("breakout_20d")),
        "breakout_55d": bool(row.get("breakout_55d")),
        "core_pass": bool(row.get("core_pass")),
        "satellite_pass": bool(row.get("satellite_pass")),
        "eps_score": int(row.get("eps_score") or 0),
        "eps_growth_yoy": _safe_float(row.get("eps_growth_yoy")),
        "ret_3m": _safe_float(row.get("ret_3m")),
        "ret_6m": _safe_float(row.get("ret_6m")),
        "ret_12m": _safe_float(row.get("ret_12m")),
        "price": _safe_float(row.get("price")),
        "sma50": _safe_float(row.get("sma50")),
        "sma150": _safe_float(row.get("sma150")),
        "sma200": _safe_float(row.get("sma200")),
        "high52w": _safe_float(row.get("high52w")),
        "low52w": _safe_float(row.get("low52w")),
        "tt_1": bool(row.get("tt_1")), "tt_2": bool(row.get("tt_2")),
        "tt_3": bool(row.get("tt_3")), "tt_4": bool(row.get("tt_4")),
        "tt_5": bool(row.get("tt_5")), "tt_6": bool(row.get("tt_6")),
        "tt_7": bool(row.get("tt_7")), "tt_8": bool(row.get("tt_8")),
        "tt_all_pass": bool(row.get("tt_all_pass")),
        "cfo_pos_5y_ratio": _safe_float(row.get("cfo_pos_5y_ratio")),
        "fcf_pos_5y_ratio": _safe_float(row.get("fcf_pos_5y_ratio")),
        "fcf_yield": _safe_float(row.get("fcf_yield")),
        "high_yield_risk": bool(row.get("high_yield_risk")),
        "core_drop_reasons": str(row.get("core_drop_reasons") or ""),
        "satellite_drop_reasons": str(row.get("satellite_drop_reasons") or ""),
    }


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f1f5f9; color: #1e293b; padding: 20px;
}
h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
.meta { font-size: 0.82rem; color: #64748b; margin-bottom: 16px; }

/* summary bar */
.summary-bar { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.stat-box {
  background: #fff; border-radius: 10px; padding: 10px 18px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08); text-align: center;
}
.stat-box .num { font-size: 1.6rem; font-weight: 700; color: #0f172a; }
.stat-box .lbl { font-size: 0.72rem; color: #64748b; margin-top: 2px; }

/* controls */
.controls { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
.btn {
  background: #3b82f6; color: #fff; border: none; border-radius: 7px;
  padding: 7px 16px; font-size: 0.83rem; cursor: pointer; font-weight: 600;
}
.btn:hover { background: #2563eb; }
.btn-outline {
  background: #fff; color: #3b82f6; border: 1.5px solid #3b82f6;
}
.btn-outline:hover { background: #eff6ff; }
.filter-label { font-size: 0.82rem; color: #64748b; }
select.filter-sel {
  border: 1.5px solid #cbd5e1; border-radius: 6px; padding: 5px 10px;
  font-size: 0.82rem; background: #fff; color: #1e293b; cursor: pointer;
}

/* table wrapper */
.tbl-wrap { overflow-x: auto; background: #fff; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.09); }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; white-space: nowrap; }
thead th {
  background: #f8fafc; border-bottom: 2px solid #e2e8f0;
  padding: 10px 10px; text-align: right; font-weight: 600; color: #475569;
  cursor: pointer; user-select: none; position: relative;
}
thead th.th-left { text-align: left; }
thead th:hover { background: #f1f5f9; }
thead th.sort-asc::after  { content: " ▲"; font-size: 0.65rem; color: #3b82f6; }
thead th.sort-desc::after { content: " ▼"; font-size: 0.65rem; color: #3b82f6; }
tbody tr.data-row { cursor: pointer; border-bottom: 1px solid #f1f5f9; transition: background .1s; }
tbody tr.data-row:hover { background: #f8fafc; }
tbody tr.data-row.expanded { background: #eff6ff; }
td { padding: 9px 10px; text-align: right; vertical-align: middle; }
td.td-left { text-align: left; }
td.code-cell { font-weight: 700; color: #1e40af; font-size: 0.88rem; }
td.name-cell { color: #1e293b; font-size: 0.82rem; max-width: 180px; overflow: hidden; text-overflow: ellipsis; }
td.sector-cell { color: #64748b; font-size: 0.75rem; }

/* colored values */
.rs-high { color: #16a34a; font-weight: 700; }
.rs-mid  { color: #ca8a04; font-weight: 700; }
.rs-low  { color: #dc2626; font-weight: 700; }
.yield-hi { color: #b45309; }  /* high yield risk */
.val-good { color: #16a34a; font-weight: 600; }
.val-warn { color: #dc2626; }

/* badges */
.badge {
  display: inline-block; font-size: 0.7rem; font-weight: 700;
  padding: 2px 7px; border-radius: 4px; white-space: nowrap;
}
.badge-core { background: #dcfce7; color: #166534; }
.badge-sat  { background: #dbeafe; color: #1e40af; }
.badge-bo   { background: #fef9c3; color: #713f12; }
.badge-pb   { background: #f3e8ff; color: #7e22ce; }
.badge-gray { background: #f1f5f9; color: #94a3b8; }

/* detail row */
tr.detail-row { display: none; }
tr.detail-row.open { display: table-row; }
tr.detail-row td { padding: 0; border-bottom: 2px solid #e2e8f0; }
.detail-inner {
  display: flex; gap: 0; padding: 14px 18px;
  background: #f8fafc; flex-wrap: wrap;
}
.detail-col { flex: 1; min-width: 180px; padding: 0 16px 0 0; }
.detail-col:last-child { padding-right: 0; }
.dl { font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
  color: #94a3b8; letter-spacing:.06em; margin-bottom:5px; margin-top:10px; }
.dl:first-child { margin-top: 0; }
.di-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.di-table th { text-align: left; color: #64748b; font-weight: 400; padding: 2px 0; width: 55%; }
.di-table td { text-align: right; font-weight: 600; padding: 2px 0; }
.tt-grid { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 4px; }
.tt-p { background:#dcfce7; color:#166534; font-size:0.68rem; font-weight:700; padding:2px 5px; border-radius:3px; }
.tt-f { background:#fee2e2; color:#991b1b; font-size:0.68rem; font-weight:700; padding:2px 5px; border-radius:3px; }

/* score bar */
.sbar-wrap {
  display:flex; align-items:center; gap:6px;
  background:#e2e8f0; border-radius:99px; height:18px;
  position:relative; overflow:hidden; min-width:80px;
}
.sbar-fill {
  background:linear-gradient(90deg,#3b82f6,#6366f1);
  border-radius:99px; height:100%; position:absolute; left:0; top:0;
}
.sbar-val { position:relative; font-size:0.7rem; font-weight:700; color:#1e293b; margin-left:auto; padding-right:6px; z-index:1; }

/* modal */
.modal-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.45); z-index: 1000;
  align-items: center; justify-content: center;
}
.modal-overlay.open { display: flex; }
.modal {
  background: #fff; border-radius: 14px; padding: 28px 32px;
  max-width: 720px; width: 95%; max-height: 90vh; overflow-y: auto;
  box-shadow: 0 8px 40px rgba(0,0,0,.18);
}
.modal h2 { font-size: 1.1rem; font-weight: 700; margin-bottom: 16px; }
.modal h3 { font-size: 0.9rem; font-weight: 700; color: #3b82f6; margin: 18px 0 8px; border-left: 3px solid #3b82f6; padding-left: 8px; }
.cond-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-bottom: 8px; }
.cond-table th { text-align: left; color: #64748b; font-weight: 600; padding: 5px 8px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }
.cond-table td { padding: 5px 8px; border-bottom: 1px solid #f1f5f9; }
.cond-table td:last-child { font-weight: 600; color: #1e293b; text-align: right; }
.modal-close {
  float: right; background: none; border: none; font-size: 1.4rem;
  cursor: pointer; color: #94a3b8; line-height: 1;
}
.modal-close:hover { color: #1e293b; }

@media (max-width: 700px) {
  .detail-inner { flex-direction: column; }
  .detail-col { min-width: 100%; }
}
"""


# ── JavaScript ────────────────────────────────────────────────────────────────

JS = """
const ROWS = __DATA__;

let sortCol = 'composite_score';
let sortDir = -1; // -1=desc, 1=asc
let filterMode = 'all'; // 'all' | 'core' | 'satellite'
let expandedIdx = null;

const fmt = (v, digits=1) => v == null ? '—' : v.toFixed(digits);
const fmtPct = (v) => v == null ? '—' : (v*100).toFixed(1)+'%';
const fmtNum = (v) => v == null ? '—' : v.toLocaleString('ja-JP', {maximumFractionDigits:0});

function rsClass(v) {
  if (v == null) return '';
  if (v >= 90) return 'rs-high';
  if (v >= 70) return 'rs-mid';
  return 'rs-low';
}

function scoreBar(score) {
  if (score == null) return '—';
  const pct = Math.min(100, Math.max(0, score / 100 * 100));
  return `<div class="sbar-wrap"><div class="sbar-fill" style="width:${pct.toFixed(1)}%"></div><span class="sbar-val">${score.toFixed(1)}</span></div>`;
}

function signalBadges(row) {
  const parts = [];
  if (row.breakout_20d) parts.push('<span class="badge badge-bo">BO20</span>');
  if (row.breakout_55d) parts.push('<span class="badge badge-bo">BO55</span>');
  if (row.first_pullback) parts.push('<span class="badge badge-pb">初押し</span>');
  return parts.length ? parts.join(' ') : '<span style="color:#cbd5e1">—</span>';
}

function coreSatBadges(row) {
  const parts = [];
  if (row.core_pass) parts.push('<span class="badge badge-core">Core</span>');
  if (row.satellite_pass) parts.push('<span class="badge badge-sat">Sat</span>');
  return parts.length ? parts.join(' ') : '<span style="color:#cbd5e1">—</span>';
}

function yieldCell(v, highRisk) {
  if (v == null) return '—';
  const s = fmtPct(v);
  if (highRisk) return `<span class="yield-hi" title="高配当リスク">${s} ⚠</span>`;
  if (v >= 0.03) return `<span class="val-good">${s}</span>`;
  return s;
}

function perCell(v) {
  if (v == null) return '—';
  const s = fmt(v, 1) + 'x';
  if (v <= 15) return `<span class="val-good">${s}</span>`;
  if (v > 30)  return `<span class="val-warn">${s}</span>`;
  return s;
}

function pbrCell(v) {
  if (v == null) return '—';
  const s = fmt(v, 2) + 'x';
  if (v <= 1.5) return `<span class="val-good">${s}</span>`;
  if (v > 3)    return `<span class="val-warn">${s}</span>`;
  return s;
}

function ttBadges(row) {
  return [1,2,3,4,5,6,7,8].map(i =>
    `<span class="${row['tt_'+i] ? 'tt-p' : 'tt-f'}">TT${i}</span>`
  ).join('');
}

function detailHtml(row) {
  const retRow = (label, v) => `<tr><th>${label}</th><td>${fmtPct(v)}</td></tr>`;
  const numRow = (label, v, digits=1) => `<tr><th>${label}</th><td>${fmt(v,digits)}</td></tr>`;
  const numFmtRow = (label, v) => `<tr><th>${label}</th><td>${fmtNum(v)}</td></tr>`;

  return `<div class="detail-inner">
  <div class="detail-col">
    <div class="dl">テクニカル</div>
    <table class="di-table">
      ${numFmtRow('株価', row.price)}
      ${numFmtRow('SMA50', row.sma50)}
      ${numFmtRow('SMA150', row.sma150)}
      ${numFmtRow('SMA200', row.sma200)}
      ${numFmtRow('52週高値', row.high52w)}
      ${numFmtRow('52週安値', row.low52w)}
    </table>
    <div class="dl" style="margin-top:10px">TT条件</div>
    <div class="tt-grid">${ttBadges(row)}</div>
  </div>
  <div class="detail-col">
    <div class="dl">リターン</div>
    <table class="di-table">
      ${retRow('3ヶ月', row.ret_3m)}
      ${retRow('6ヶ月', row.ret_6m)}
      ${retRow('12ヶ月', row.ret_12m)}
    </table>
    <div class="dl" style="margin-top:10px">EPS加速</div>
    <table class="di-table">
      <tr><th>スコア</th><td>${row.eps_score}/10</td></tr>
      <tr><th>YoY成長</th><td>${fmtPct(row.eps_growth_yoy)}</td></tr>
    </table>
  </div>
  <div class="detail-col">
    <div class="dl">ファンダメンタルズ</div>
    <table class="di-table">
      ${numRow('非減配年数', row.non_cut_years, 0)}
      <tr><th>CFO黒字率(5y)</th><td>${fmtPct(row.cfo_pos_5y_ratio)}</td></tr>
      <tr><th>FCF黒字率(5y)</th><td>${fmtPct(row.fcf_pos_5y_ratio)}</td></tr>
      <tr><th>FCF配当性向(3y)</th><td>${fmtPct(row.fcf_payout_3y)}</td></tr>
      <tr><th>FCF利回り</th><td>${fmtPct(row.fcf_yield)}</td></tr>
    </table>
    ${row.core_drop_reasons ? `<div style="margin-top:8px;font-size:0.72rem;color:#b45309;background:#fef3c7;padding:4px 8px;border-radius:6px">Core落ち: ${row.core_drop_reasons}</div>` : ''}
    ${row.satellite_drop_reasons ? `<div style="margin-top:4px;font-size:0.72rem;color:#1e40af;background:#dbeafe;padding:4px 8px;border-radius:6px">Sat落ち: ${row.satellite_drop_reasons}</div>` : ''}
  </div>
</div>`;
}

function getFilteredRows() {
  if (filterMode === 'core') return ROWS.filter(r => r.core_pass);
  if (filterMode === 'satellite') return ROWS.filter(r => r.satellite_pass);
  return ROWS;
}

function render() {
  const rows = getFilteredRows();
  const sorted = [...rows].sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    // booleans: true=1, false=0
    if (typeof va === 'boolean') va = va ? 1 : 0;
    if (typeof vb === 'boolean') vb = vb ? 1 : 0;
    if (va == null) va = sortDir === -1 ? -Infinity : Infinity;
    if (vb == null) vb = sortDir === -1 ? -Infinity : Infinity;
    if (va < vb) return sortDir;
    if (va > vb) return -sortDir;
    return 0;
  });

  const tbody = document.getElementById('tbl-body');
  tbody.innerHTML = '';

  sorted.forEach((row, i) => {
    const tr = document.createElement('tr');
    tr.className = 'data-row';
    tr.dataset.idx = i;
    tr.innerHTML = `
      <td class="td-left" style="color:#94a3b8;font-size:0.8rem">${i+1}</td>
      <td class="td-left code-cell">${row.code_disp}</td>
      <td class="td-left name-cell" title="${row.name}">${row.name}</td>
      <td class="td-left sector-cell">${row.sector}</td>
      <td>${scoreBar(row.composite_score)}</td>
      <td class="${rsClass(row.rs_percentile)}">${fmt(row.rs_percentile,1)}</td>
      <td>${yieldCell(row.dividend_yield_fwd, row.high_yield_risk)}</td>
      <td>${perCell(row.per_fwd)}</td>
      <td>${pbrCell(row.pbr)}</td>
      <td>${row.non_cut_years != null ? row.non_cut_years.toFixed(0)+'年' : '—'}</td>
      <td>${fmtPct(row.fcf_payout_3y)}</td>
      <td class="td-left">${signalBadges(row)}</td>
      <td class="td-left">${coreSatBadges(row)}</td>
    `;
    tr.addEventListener('click', () => toggleDetail(i, row, tr));
    tbody.appendChild(tr);

    // detail row
    const dtr = document.createElement('tr');
    dtr.className = 'detail-row';
    dtr.dataset.detail = i;
    const dtd = document.createElement('td');
    dtd.colSpan = 13;
    dtd.innerHTML = detailHtml(row);
    dtr.appendChild(dtd);
    tbody.appendChild(dtr);
  });

  updateSortHeaders();
}

function toggleDetail(idx, row, tr) {
  const dtr = document.querySelector(`tr[data-detail="${idx}"]`);
  if (!dtr) return;
  const isOpen = dtr.classList.contains('open');
  // close all
  document.querySelectorAll('tr.detail-row.open').forEach(el => el.classList.remove('open'));
  document.querySelectorAll('tr.data-row.expanded').forEach(el => el.classList.remove('expanded'));
  if (!isOpen) {
    dtr.classList.add('open');
    tr.classList.add('expanded');
  }
}

function updateSortHeaders() {
  document.querySelectorAll('thead th[data-col]').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.col === sortCol) {
      th.classList.add(sortDir === -1 ? 'sort-desc' : 'sort-asc');
    }
  });
}

function setSort(col) {
  if (sortCol === col) {
    sortDir = -sortDir;
  } else {
    sortCol = col;
    sortDir = -1;
  }
  render();
}

function setFilter(mode) {
  filterMode = mode;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active-filter'));
  document.getElementById('filter-' + mode).classList.add('active-filter');
  render();
}

// modal
function openModal() { document.getElementById('cond-modal').classList.add('open'); }
function closeModal() { document.getElementById('cond-modal').classList.remove('open'); }

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('thead th[data-col]').forEach(th => {
    th.addEventListener('click', () => setSort(th.dataset.col));
  });
  document.getElementById('cond-modal').addEventListener('click', e => {
    if (e.target === document.getElementById('cond-modal')) closeModal();
  });
  render();
});
"""


# ── write_html ────────────────────────────────────────────────────────────────

def write_html(df: pd.DataFrame, asof: date, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "report.html"

    rows_data = [_row_to_dict(row) for _, row in df.iterrows()]
    rows_json = json.dumps(rows_data, ensure_ascii=False)

    n = len(df)
    avg_rs = df["rs_percentile"].mean() if "rs_percentile" in df.columns else float("nan")
    bo_count = int(df["breakout_20d"].sum()) if "breakout_20d" in df.columns else 0
    pb_count = int(df["first_pullback"].sum()) if "first_pullback" in df.columns else 0
    n_core = int(df["core_pass"].sum()) if "core_pass" in df.columns else 0
    n_sat = int(df["satellite_pass"].sum()) if "satellite_pass" in df.columns else 0

    js_code = JS.replace("__DATA__", rows_json)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>スクリーニング結果 {asof}</title>
<style>
{CSS}
.filter-btn {{ background:#fff; color:#475569; border:1.5px solid #cbd5e1; border-radius:7px; padding:6px 14px; font-size:0.82rem; cursor:pointer; font-weight:600; }}
.filter-btn:hover {{ background:#f1f5f9; }}
.filter-btn.active-filter {{ background:#3b82f6; color:#fff; border-color:#3b82f6; }}
</style>
</head>
<body>
<h1>増配バリュー × ミネルヴィニ スクリーナー</h1>
<div class="meta">スクリーニング日: {asof} &nbsp;|&nbsp; 生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="summary-bar">
  <div class="stat-box"><div class="num">{n}</div><div class="lbl">候補銘柄数</div></div>
  <div class="stat-box"><div class="num">{n_core}</div><div class="lbl">Core通過</div></div>
  <div class="stat-box"><div class="num">{n_sat}</div><div class="lbl">Satellite通過</div></div>
  <div class="stat-box"><div class="num">{avg_rs:.1f}</div><div class="lbl">平均RSパーセンタイル</div></div>
  <div class="stat-box"><div class="num">{bo_count}</div><div class="lbl">ブレイクアウト(20日)</div></div>
  <div class="stat-box"><div class="num">{pb_count}</div><div class="lbl">初押しシグナル</div></div>
</div>

<div class="controls">
  <button class="filter-btn active-filter" id="filter-all"    onclick="setFilter('all')">全候補</button>
  <button class="filter-btn" id="filter-core"      onclick="setFilter('core')">Core ({n_core})</button>
  <button class="filter-btn" id="filter-satellite" onclick="setFilter('satellite')">Satellite ({n_sat})</button>
  <span style="flex:1"></span>
  <button class="btn btn-outline" onclick="openModal()">📋 スクリーニング条件</button>
</div>

<div class="tbl-wrap">
<table>
<thead>
<tr>
  <th class="th-left" style="width:36px">#</th>
  <th class="th-left" style="width:52px">コード</th>
  <th class="th-left" style="min-width:120px">銘柄名</th>
  <th class="th-left" style="min-width:80px">セクター</th>
  <th data-col="composite_score"  style="min-width:100px">スコア</th>
  <th data-col="rs_percentile"    style="min-width:64px">RS%ile</th>
  <th data-col="dividend_yield_fwd" style="min-width:72px">配当利回り</th>
  <th data-col="per_fwd"          style="min-width:56px">PER</th>
  <th data-col="pbr"              style="min-width:56px">PBR</th>
  <th data-col="non_cut_years"    style="min-width:60px">非減配年</th>
  <th data-col="fcf_payout_3y"    style="min-width:80px">FCF配当性向</th>
  <th data-col="first_pullback" class="th-left" style="min-width:100px">シグナル</th>
  <th class="th-left" style="min-width:90px">判定</th>
</tr>
</thead>
<tbody id="tbl-body"></tbody>
</table>
</div>

<!-- Conditions modal -->
<div class="modal-overlay" id="cond-modal">
<div class="modal">
  <button class="modal-close" onclick="closeModal()">✕</button>
  <h2>スクリーニング条件</h2>

  <h3>Core モード（増配バリュー）</h3>
  <table class="cond-table">
    <tr><th>条件</th><th>閾値</th></tr>
    <tr><td>非減配年数（連続）</td><td>≥ 2年</td></tr>
    <tr><td>CFO黒字率（過去5年）</td><td>≥ 80%</td></tr>
    <tr><td>FCF黒字率（過去5年）</td><td>≥ 60%</td></tr>
    <tr><td>FCF配当性向（3年累計）</td><td>≤ 70%（景気敏感業種 ≤ 60%）</td></tr>
    <tr><td>FCF正常（直近3年でFCF&gt;0）</td><td>2年以上</td></tr>
    <tr><td>配当利回り（予想）</td><td>≥ 3%</td></tr>
    <tr><td>バリュー条件（いずれか1つ）</td><td>PER≤15 or PBR≤1.5 or FCF利回り≥4%<br><span style="color:#64748b;font-size:0.78rem">or セクター内PERで下位30%</span></td></tr>
    <tr><td colspan="2" style="color:#3b82f6;font-weight:600;padding-top:8px">トレンド条件（緩）</td></tr>
    <tr><td>株価 &gt; SMA200</td><td>—</td></tr>
    <tr><td>SMA50 &gt; SMA200</td><td>—</td></tr>
    <tr><td>SMA200が上昇トレンド</td><td>20日前比で上昇</td></tr>
    <tr><td>52週安値からの水準</td><td>≥ ×1.20</td></tr>
    <tr><td>52週高値からの乖離</td><td>≥ ×0.60（高値の60%以上）</td></tr>
    <tr><td>RSパーセンタイル</td><td>≥ 70</td></tr>
  </table>

  <h3>Satellite モード（ミネルヴィニ＋オニール）</h3>
  <table class="cond-table">
    <tr><th>条件</th><th>閾値</th></tr>
    <tr><td>Minervini TT条件</td><td>全8条件クリア</td></tr>
    <tr><td>配当利回り（予想）</td><td>≥ 2%</td></tr>
    <tr><td>O'Neil EPSスコア</td><td>≥ 1（YoY成長あり）</td></tr>
  </table>

  <h3>Minervini トレンドテンプレート（TT）</h3>
  <table class="cond-table">
    <tr><th>番号</th><th>条件</th></tr>
    <tr><td>TT1</td><td>株価 &gt; SMA150 かつ SMA200</td></tr>
    <tr><td>TT2</td><td>SMA150 &gt; SMA200</td></tr>
    <tr><td>TT3</td><td>SMA200が20日前比で上昇</td></tr>
    <tr><td>TT4</td><td>SMA50 &gt; SMA150 かつ SMA200</td></tr>
    <tr><td>TT5</td><td>株価 &gt; SMA50</td></tr>
    <tr><td>TT6</td><td>株価 ≥ 52週安値 × 1.30</td></tr>
    <tr><td>TT7</td><td>株価 ≥ 52週高値 × 0.75</td></tr>
    <tr><td>TT8</td><td>RSパーセンタイル ≥ 70</td></tr>
  </table>

  <h3>コンポジットスコア（ソート基準）</h3>
  <table class="cond-table">
    <tr><th>要素</th><th>ウェイト</th></tr>
    <tr><td>RSパーセンタイル</td><td>× 0.5</td></tr>
    <tr><td>EPSスコア（0〜10）</td><td>× 3</td></tr>
    <tr><td>ブレイクアウト（20日）</td><td>+5点</td></tr>
    <tr><td>初押しシグナル</td><td>+3点</td></tr>
    <tr><td>非減配年数（最大10年キャップ）</td><td>× 0.5</td></tr>
    <tr><td>配当利回り ≥ 4%</td><td>+3点</td></tr>
  </table>

  <h3>景気敏感業種（FCF配当性向を60%上限に厳格化）</h3>
  <div style="font-size:0.82rem;color:#64748b;margin-top:4px">
    鉄鋼 / 非鉄金属 / 石油・石炭製品 / 化学 / 海運業 / 空運業 / 鉱業 / 建設業
  </div>
</div>
</div>

<script>
{js_code}
</script>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    return out_path
