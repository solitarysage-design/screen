# 増配バリュー × ミネルヴィニ スクリーナー

日本株を対象に **増配バリュー** と **ミネルヴィニ トレンドテンプレート (TT)** を組み合わせたスクリーナーです。

## データソース

| データ | 取得元 |
|--------|--------|
| 銘柄一覧 | J-Quants Light (`get_listed_info`) |
| 日足株価（調整後） | J-Quants Light → yfinance フォールバック |
| 財務・EPS | J-Quants Light (`get_statements`) → yfinance |
| CF / CapEx / 配当 | yfinance |
| TOPIX | yfinance (`^TOPX` / `1306.T`) |

## セットアップ

```bash
# Python 3.10+
pip install -e ".[dev]"

# J-Quants API トークンを環境変数に設定
export JQUANTS_MAIL_ADDRESS=your@email.com
export JQUANTS_PASSWORD=yourpassword
```

## 使い方

```bash
# 基本実行（プライム市場、利回り3%以上）
screen --asof 2025-01-17 --universe prime --min_yield 0.03

# 全市場、詳細説明付き
screen --asof 2025-01-17 --universe all --explain

# TOPIX相対RS、エクスポート無効
screen --asof 2025-01-17 --rs_mode topix --no-export

# カスタム銘柄リスト
screen --asof 2025-01-17 --custom_csv ./my_watchlist.csv
```

## オプション

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--asof` | 必須 | スクリーニング日 (YYYY-MM-DD) |
| `--universe` | `all` | `prime` / `standard` / `growth` / `all` |
| `--min_yield` | `0.03` | 最低配当利回り (3%) |
| `--lookback_days` | `420` | 価格取得期間（日数） |
| `--rs_mode` | `universe_percentile` | `universe_percentile` / `topix` |
| `--export` | `True` | CSV・watchlist出力 |
| `--explain` | `False` | 銘柄別条件詳細表示 |
| `--unknown_policy` | `exclude` | データ不明時の扱い (`include`/`exclude`) |

## スクリーニング条件

### ミネルヴィニ トレンドテンプレート（全8条件 AND）

| # | 条件 |
|---|------|
| TT-1 | 株価 > SMA150 かつ SMA200 |
| TT-2 | SMA150 > SMA200 |
| TT-3 | SMA200 が上昇トレンド（20営業日前比） |
| TT-4 | SMA50 > SMA150 かつ SMA200 |
| TT-5 | 株価 > SMA50 |
| TT-6 | 株価 ≥ 52週安値 × 1.30 |
| TT-7 | 株価 ≥ 52週高値 × 0.75 |
| TT-8 | RSパーセンタイル ≥ 70 |

### 増配バリュー ハードフィルター

| # | 条件 |
|---|------|
| A | 非減配年数 ≥ 3年 |
| B | 過去5年でCFOがプラスの年 ≥ 60% |
| C | 過去5年でFCFがプラスの年 ≥ 60% |
| D | FCF配当性向3年平均 ≤ 70%（景気敏感株60%） |
| E | 配当利回り ≥ 指定値（デフォルト3%） |

## 出力

```
C:/Users/solit/projects/screen/output/
├── candidates.csv        # スクリーニング通過銘柄（スコア順）
└── moomoo_watchlist.txt  # moomoo形式ウォッチリスト
```

## CI/CD (GitHub Actions)

毎週金曜 16:00 JST に自動実行されます。手動トリガーも可能です。

### Secrets 設定

リポジトリの Settings > Secrets and variables > Actions で以下を設定:

| Secret | 用途 | 必須 |
|--------|------|------|
| `JQUANTS_MAIL_ADDRESS` | J-Quants API ログインメール | Yes |
| `JQUANTS_PASSWORD` | J-Quants API パスワード | Yes |
| `SLACK_WEBHOOK_URL` | Slack 通知 webhook | No |

### 手動実行

1. GitHub リポジトリの **Actions** タブを開く
2. 左メニューから **Weekly Stock Screener** を選択
3. **Run workflow** をクリック
4. (任意) スクリーニング日を入力 (YYYY-MM-DD)
5. **Run workflow** で実行

### Artifacts の確認

1. **Actions** タブ > 対象の Run をクリック
2. **Artifacts** セクションから `weekly-screen-output` をダウンロード

### ローカル実行

```bash
# 今日の日付でスクリーニング
python -m screening.run_weekly

# 日付指定
python -m screening.run_weekly --asof 2026-03-03

# 環境変数で日付指定
SCREEN_ASOF_DATE=2026-03-03 python -m screening.run_weekly
```

出力先: `outputs/YYYY-MM-DD/` (最新は `outputs/latest/` にもコピー)

### 保有株データの更新

`data/holdings.csv` を更新してリポジトリに push すれば、次回の CI 実行から反映されます。

## テスト

```bash
pytest tests/ -v
```

## キャッシュ

価格データは `~/.cache/screen/` にキャッシュされます（価格6時間・財務24時間）。

## プロジェクト構成

```
screen/
├── config.py                       # 全定数・閾値
├── data/
│   ├── cache.py                    # diskcache TTLラッパー
│   ├── jquants_client.py           # JQ APIクライアント（retry付き）
│   ├── prices.py                   # 日足取得
│   ├── fundamentals.py             # 財務・配当取得
│   └── universe.py                 # 銘柄一覧
├── features/
│   ├── technicals.py               # SMA/52週高安値
│   ├── relative_strength.py        # RSスコア
│   ├── breakouts.py                # ブレイク/初押し
│   └── fundamentals_metrics.py     # CF・配当指標
├── screens/
│   ├── minervini_gate.py           # TT条件
│   ├── hard_dividend_value.py      # 増配バリューフィルター
│   └── oniel_accel.py              # EPSスコア
└── cli.py                          # CLIエントリポイント
```
