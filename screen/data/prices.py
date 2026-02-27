"""Daily price data retrieval: J-Quants bulk primary, yfinance fallback.

Strategy:
1. J-Quants get_price_range() で全銘柄を一括取得（高速）
   - プランの対応日付範囲を自動検出・キャップ
   - J-Quants カバレッジ終了日 < 要求終了日 の場合、不足期間を yfinance で補完
2. J-Quants に存在しない銘柄は yfinance バッチ取得（150銘柄単位）
3. 欠損率 > MAX_MISSING_PRICE_RATIO の銘柄は除外リストへ
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from screen.config import CACHE_TTL_PRICES, MAX_MISSING_PRICE_RATIO
from screen.data.jquants_client import get_client

logger = logging.getLogger(__name__)

# J-Quants Light プランの利用可能日付範囲（エラーから自動検出）
_JQ_AVAILABLE_FROM: date | None = None
_JQ_AVAILABLE_TO: date | None = None

_YF_BATCH_SIZE = 150  # yfinance バッチサイズ


def _parse_available_from(error_msg: str) -> date | None:
    """エラーメッセージから利用可能開始日を抽出する。"""
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*~", str(error_msg))
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    return None


def _parse_available_to(error_msg: str) -> date | None:
    """エラーメッセージから利用可能終了日を抽出する。"""
    m = re.search(r"~\s*(\d{4}-\d{2}-\d{2})", str(error_msg))
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    return None


def _fetch_jq_bulk(start: date, end: date) -> tuple[pd.DataFrame | None, date | None]:
    """J-Quants で全銘柄の価格を一括取得。

    プランの日付範囲制限を自動検出してリトライする。

    Returns:
        (DataFrame or None, effective_end_date)
        effective_end_date: 実際に取得できた末尾日付（プラン上限が適用された場合はそれ以前）
    """
    global _JQ_AVAILABLE_FROM, _JQ_AVAILABLE_TO
    client = get_client()

    effective_start = start
    effective_end = end

    # キャッシュ済みの制限を適用
    if _JQ_AVAILABLE_FROM and effective_start < _JQ_AVAILABLE_FROM:
        effective_start = _JQ_AVAILABLE_FROM
    if _JQ_AVAILABLE_TO and effective_end > _JQ_AVAILABLE_TO:
        effective_end = _JQ_AVAILABLE_TO
        logger.info("J-Quants: プラン上限により終了日を %s にキャップ", effective_end)

    if effective_start > effective_end:
        logger.warning("J-Quants: 有効日付範囲なし (%s > %s)", effective_start, effective_end)
        return None, None

    try:
        df = client.get_price_range(
            start_dt=effective_start.strftime("%Y%m%d"),
            end_dt=effective_end.strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            return None, effective_end
        logger.info(
            "J-Quants bulk fetch: %s → %s, %d rows, %d codes",
            effective_start, effective_end, len(df), df["Code"].nunique(),
        )
        return df, effective_end

    except Exception as exc:
        err_str = str(exc)
        avail_from = _parse_available_from(err_str)
        avail_to = _parse_available_to(err_str)

        changed = False
        if avail_from and avail_from > effective_start:
            logger.warning("J-Quants: 開始日を %s に調整", avail_from)
            _JQ_AVAILABLE_FROM = avail_from
            effective_start = avail_from
            changed = True
        if avail_to and avail_to < effective_end:
            logger.warning("J-Quants: 終了日を %s にキャップ（プラン上限）", avail_to)
            _JQ_AVAILABLE_TO = avail_to
            effective_end = avail_to
            changed = True

        if changed and effective_start <= effective_end:
            try:
                df = client.get_price_range(
                    start_dt=effective_start.strftime("%Y%m%d"),
                    end_dt=effective_end.strftime("%Y%m%d"),
                )
                if df is not None and not df.empty:
                    logger.info(
                        "J-Quants retry OK: %s → %s, %d rows, %d codes",
                        effective_start, effective_end, len(df), df["Code"].nunique(),
                    )
                    return df, effective_end
            except Exception as exc2:
                logger.error("J-Quants retry failed: %s", exc2)
        else:
            logger.error("J-Quants bulk fetch failed: %s", exc)

    return None, None


def _normalise_jq(df: pd.DataFrame) -> pd.DataFrame:
    """J-Quants の列を正規化して [Date, Code, Close, Volume] に絞り込む。
    AdjustmentClose / AdjustmentVolume を優先使用。
    """
    cols = df.columns.tolist()

    close_col = next(
        (c for c in cols if c == "AdjustmentClose"),
        next((c for c in cols if c.lower() == "close"), None),
    )
    vol_col = next(
        (c for c in cols if c == "AdjustmentVolume"),
        next((c for c in cols if c.lower() == "volume"), None),
    )
    date_col = next((c for c in cols if c.lower() in ("date", "tradingdate")), None)
    code_col = next((c for c in cols if c.lower() == "code"), None)

    if close_col is None or date_col is None or code_col is None:
        logger.error("J-Quants normalise: 必須列が見つからない. columns=%s", cols)
        return pd.DataFrame(columns=["Date", "Code", "Close", "Volume"])

    result = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col]),
        "Code": df[code_col].astype(str),
        "Close": pd.to_numeric(df[close_col], errors="coerce"),
        "Volume": pd.to_numeric(df[vol_col], errors="coerce") if vol_col else float("nan"),
    })
    return result.dropna(subset=["Close"])


def _jq_to_yf_ticker(code: str) -> str:
    """J-Quants コード (5桁 末尾0) → yfinance ティッカー (4桁 + .T)。
    例: "72030" → "7203.T"、"1234A" → "1234A.T"（英字入りは不変）
    """
    if len(code) == 5 and code.endswith("0") and code[:4].isdigit():
        return f"{code[:4]}.T"
    return f"{code}.T"


def _fetch_yf_batch(
    codes: list[str],
    start: date,
    end: date,
    batch_size: int = _YF_BATCH_SIZE,
) -> dict[str, pd.DataFrame]:
    """yfinance で複数銘柄を一括取得する（batch_size 件ずつ）。"""
    if not codes:
        return {}

    tickers = [_jq_to_yf_ticker(c) for c in codes]
    ticker_to_code = {_jq_to_yf_ticker(c): c for c in codes}
    all_frames: dict[str, pd.DataFrame] = {}

    n_batches = (len(tickers) + batch_size - 1) // batch_size
    start_str = start.strftime("%Y-%m-%d")
    end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")

    for batch_idx in range(n_batches):
        batch_tickers = tickers[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_codes = codes[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        try:
            raw = yf.download(
                batch_tickers,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as exc:
            logger.warning("yfinance batch %d/%d 失敗: %s", batch_idx + 1, n_batches, exc)
            continue

        if raw is None or raw.empty:
            continue

        got = 0
        if isinstance(raw.columns, pd.MultiIndex):
            top_level = raw.columns.get_level_values(0).unique().tolist()
            for ticker in batch_tickers:
                code = ticker_to_code.get(ticker)
                if code is None or ticker not in top_level:
                    continue
                try:
                    ticker_df = raw[ticker].reset_index()
                    close_col = next(
                        (c for c in ticker_df.columns if str(c).lower() == "close"), None
                    )
                    date_col = next(
                        (c for c in ticker_df.columns if str(c).lower() == "date"), None
                    )
                    vol_col = next(
                        (c for c in ticker_df.columns if str(c).lower() == "volume"), None
                    )
                    if not close_col or not date_col:
                        continue
                    df = pd.DataFrame({
                        "Date": pd.to_datetime(ticker_df[date_col]),
                        "Code": code,
                        "Close": pd.to_numeric(ticker_df[close_col], errors="coerce"),
                        "Volume": pd.to_numeric(ticker_df[vol_col], errors="coerce")
                        if vol_col else float("nan"),
                    }).dropna(subset=["Close"])
                    if not df.empty:
                        all_frames[code] = df.sort_values("Date").reset_index(drop=True)
                        got += 1
                except Exception:
                    continue
        elif len(batch_tickers) == 1:
            # 1件のみの場合は MultiIndex にならない
            ticker = batch_tickers[0]
            code = ticker_to_code.get(ticker)
            if code:
                try:
                    ticker_df = raw.reset_index()
                    close_col = next(
                        (c for c in ticker_df.columns if str(c).lower() == "close"), None
                    )
                    date_col = next(
                        (c for c in ticker_df.columns if str(c).lower() == "date"), None
                    )
                    vol_col = next(
                        (c for c in ticker_df.columns if str(c).lower() == "volume"), None
                    )
                    if close_col and date_col:
                        df = pd.DataFrame({
                            "Date": pd.to_datetime(ticker_df[date_col]),
                            "Code": code,
                            "Close": pd.to_numeric(ticker_df[close_col], errors="coerce"),
                            "Volume": pd.to_numeric(ticker_df[vol_col], errors="coerce")
                            if vol_col else float("nan"),
                        }).dropna(subset=["Close"])
                        if not df.empty:
                            all_frames[code] = df.sort_values("Date").reset_index(drop=True)
                            got = 1
                except Exception:
                    pass

        logger.info(
            "yfinance batch %d/%d: %d tickers → %d 件取得",
            batch_idx + 1, n_batches, len(batch_tickers), got,
        )

    return all_frames


def _fetch_yf_single(code: str, start: date, end: date) -> pd.DataFrame | None:
    """yfinance で単一銘柄を取得する（バッチで取れなかった場合の個別フォールバック）。"""
    ticker = _jq_to_yf_ticker(code)
    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.debug("yfinance single failed for %s: %s", ticker, exc)
        return None

    if raw is None or raw.empty:
        return None

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(filter(None, map(str, c))) for c in raw.columns]

    close_col = next((c for c in raw.columns if "close" in c.lower()), None)
    vol_col = next((c for c in raw.columns if "volume" in c.lower()), None)
    date_col = next((c for c in raw.columns if "date" in c.lower()), None)

    if not close_col or not date_col:
        return None

    result = pd.DataFrame({
        "Date": pd.to_datetime(raw[date_col]),
        "Code": code,
        "Close": pd.to_numeric(raw[close_col], errors="coerce"),
        "Volume": pd.to_numeric(raw[vol_col], errors="coerce") if vol_col else float("nan"),
    })
    return result.dropna(subset=["Close"]).sort_values("Date")


def get_prices(
    codes: list[str],
    start: date,
    end: date,
) -> tuple[pd.DataFrame, list[str]]:
    """全銘柄の調整後日足データを取得する。

    戦略:
    1. J-Quants get_price_range() で全銘柄を一括取得（高速）
       - プラン上限日が要求終了日より前の場合、残り期間を yfinance バッチで補完
    2. J-Quants に存在しない銘柄は yfinance バッチ取得
    3. それでも取れない銘柄は個別取得
    4. 欠損率 > MAX_MISSING_PRICE_RATIO の銘柄は除外リストへ

    Returns:
        (prices_df, excluded_codes)
    """
    codes_set = set(codes)
    excluded: list[str] = []
    expected_days = max(1, (end - start).days * 5 // 7)

    # --- Step 1: J-Quants 一括取得 ---
    jq_df, jq_effective_end = _fetch_jq_bulk(start, end)
    jq_frames: dict[str, pd.DataFrame] = {}

    if jq_df is not None:
        jq_norm = _normalise_jq(jq_df)
        jq_norm = jq_norm[jq_norm["Code"].isin(codes_set)]
        for code, grp in jq_norm.groupby("Code"):
            jq_frames[str(code)] = grp.sort_values("Date").reset_index(drop=True)
        logger.info("J-Quants: %d / %d codes have data", len(jq_frames), len(codes))
    else:
        logger.warning("J-Quants 一括取得失敗。全銘柄 yfinance フォールバック。")

    # --- Step 2: J-Quants カバレッジ外の期間を yfinance バッチで補完 ---
    # （例: JQ データが 2025-12-03 まで、要求が 2026-02-25 まで）
    if jq_frames and jq_effective_end is not None and jq_effective_end < end:
        supplement_start = jq_effective_end + timedelta(days=1)
        logger.info(
            "J-Quants カバレッジ外期間 %s → %s を yfinance バッチで補完中 (%d銘柄)...",
            supplement_start, end, len(codes),
        )
        supplement_frames = _fetch_yf_batch(codes, supplement_start, end)
        supplemented = 0
        for code, supp_df in supplement_frames.items():
            if code in jq_frames:
                combined = pd.concat(
                    [jq_frames[code], supp_df], ignore_index=True
                ).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
                jq_frames[code] = combined
            else:
                jq_frames[code] = supp_df
            supplemented += 1
        logger.info("補完完了: %d銘柄に最新データを追記", supplemented)

    # --- Step 3: JQ で全く取れなかった銘柄を yfinance バッチで補完 ---
    missing_from_jq = [c for c in codes if c not in jq_frames]
    yf_frames: dict[str, pd.DataFrame] = {}

    if missing_from_jq:
        logger.info("yfinance バッチフォールバック: %d 銘柄", len(missing_from_jq))
        yf_frames = _fetch_yf_batch(missing_from_jq, start, end)

        # バッチで取れなかった銘柄を個別フォールバック
        still_missing = [c for c in missing_from_jq if c not in yf_frames]
        if still_missing:
            logger.info("yfinance 個別フォールバック: %d 銘柄", len(still_missing))
            for code in still_missing:
                df = _fetch_yf_single(code, start, end)
                if df is not None and not df.empty:
                    yf_frames[code] = df
                else:
                    logger.debug("%s: yfinance も空", code)

    # --- Step 4: 結合・欠損率チェック ---
    all_frames: list[pd.DataFrame] = []

    for code in codes:
        if code in jq_frames:
            df = jq_frames[code]
            src = "jquants"
        elif code in yf_frames:
            df = yf_frames[code]
            src = "yfinance"
        else:
            logger.debug("%s: データなし → 除外", code)
            excluded.append(code)
            continue

        missing_ratio = 1 - len(df) / expected_days
        if missing_ratio > MAX_MISSING_PRICE_RATIO:
            logger.debug(
                "%s: 欠損率 %.1f%% > %.0f%% → 除外 (%s)",
                code, missing_ratio * 100, MAX_MISSING_PRICE_RATIO * 100, src,
            )
            excluded.append(code)
            continue

        df = df.copy()
        df["Code"] = code
        df["source"] = src
        all_frames.append(df)

    if excluded:
        logger.info("価格除外銘柄: %d件", len(excluded))

    if not all_frames:
        return pd.DataFrame(columns=["Date", "Code", "Close", "Volume"]), excluded

    prices = pd.concat(all_frames, ignore_index=True)
    prices["Date"] = pd.to_datetime(prices["Date"])
    logger.info(
        "価格取得完了: %d 銘柄, %d 行 (除外: %d)",
        prices["Code"].nunique(), len(prices), len(excluded),
    )
    return prices.sort_values(["Code", "Date"]).reset_index(drop=True), excluded


def get_topix(start: date, end: date) -> pd.DataFrame | None:
    """TOPIX 指数価格を取得する。^TOPX → 1306.T の順でフォールバック。"""
    for ticker in ["^TOPX", "1306.T"]:
        try:
            raw = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if raw is not None and not raw.empty:
                raw = raw.reset_index()
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = ["_".join(filter(None, map(str, c))) for c in raw.columns]
                close_col = next((c for c in raw.columns if "close" in c.lower()), None)
                date_col = next((c for c in raw.columns if "date" in c.lower()), None)
                if close_col and date_col:
                    result = pd.DataFrame({
                        "Date": pd.to_datetime(raw[date_col]),
                        "Close": pd.to_numeric(raw[close_col], errors="coerce"),
                    }).dropna().sort_values("Date")
                    logger.info("TOPIX: %s (%d rows)", ticker, len(result))
                    return result
        except Exception as exc:
            logger.warning("TOPIX %s 失敗: %s", ticker, exc)

    logger.warning("TOPIX 取得失敗")
    return None
