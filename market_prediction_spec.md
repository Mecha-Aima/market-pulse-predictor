# Implementation Spec: Real-Time Market Movement Prediction System

**Version:** 1.0  
**Approach:** Test-Driven Development (TDD)  
**Audience:** AI coding agent — build this system exactly as specified, phase by phase.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository & Project Structure](#2-repository--project-structure)
3. [Technology Stack](#3-technology-stack)
4. [Environment & Secrets Management](#4-environment--secrets-management)
5. [Phase 0 — Infrastructure Setup](#phase-0--infrastructure-setup)
6. [Phase 1 — Data Ingestion](#phase-1--data-ingestion)
7. [Phase 2 — Sentiment Analysis](#phase-2--sentiment-analysis)
8. [Phase 3 — Time-Series Feature Construction](#phase-3--time-series-feature-construction)
9. [Phase 4 — Model Training & Experiment Tracking](#phase-4--model-training--experiment-tracking)
10. [Phase 5 — REST API](#phase-5--rest-api)
11. [Phase 6 — Frontend Dashboard](#phase-6--frontend-dashboard)
12. [Phase 7 — Airflow Orchestration](#phase-7--airflow-orchestration)
13. [Phase 8 — CI/CD with GitHub Actions](#phase-8--cicd-with-github-actions)
14. [Phase 9 — Docker & AWS EC2 Deployment](#phase-9--docker--aws-ec2-deployment)
15. [Data Schema Reference](#data-schema-reference)
16. [Testing Strategy](#testing-strategy)
17. [MLflow Experiment Convention](#mlflow-experiment-convention)
18. [DVC Data Versioning Convention](#dvc-data-versioning-convention)
19. [Known Constraints & Mitigations](#known-constraints--mitigations)

---

## 1. System Overview

This system ingests live financial text data from multiple public sources, classifies sentiment, builds a rolling time-series, and uses three sequential deep learning models (SimpleRNN, LSTM, GRU) to predict:

- **Market direction**: Up / Down / Flat (classification)
- **Price movement trend**: Smoothed next-period return (regression)
- **Volatility spike**: Binary flag — spike or no spike (classification)

The system runs continuously. Airflow orchestrates the pipeline. All data versions are tracked by DVC. All training experiments are logged by MLflow. The trained model is served via FastAPI. A Streamlit dashboard presents live predictions and experiment results.

### Pipeline at a Glance

```
[Data Sources]
  Yahoo Finance API
  Reuters RSS Feeds
  Reddit (PRAW)
  Twitter/X (snscrape fallback)
        |
        v
[Airflow DAG: ingestion_dag]
  Raw text + price data → data/raw/
        |
        v
[Airflow DAG: sentiment_dag]
  Sentiment labels (pos/neg/neu) → data/processed/
        |
        v
[Airflow DAG: feature_dag]
  Aggregated time-series features → data/features/
        |
        v
[Airflow DAG: training_dag]   ←→  MLflow Tracking Server
  Train RNN / LSTM / GRU
  Evaluate + register best model
        |
        v
[FastAPI REST API]  ←→  Streamlit Dashboard
  /predict  /results  /health
        |
        v
[Docker + AWS EC2]
```

---

## 2. Repository & Project Structure

Repository name: `market-pulse-predictor` (use this exact name)

```
market-pulse-predictor/
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint, test on every PR
│       └── cd.yml              # Build + deploy on merge to main
│
├── airflow/
│   ├── dags/
│   │   ├── ingestion_dag.py
│   │   ├── sentiment_dag.py
│   │   ├── feature_dag.py
│   │   └── training_dag.py
│   └── plugins/               # Empty, reserved for custom operators
│
├── data/
│   ├── raw/                   # DVC-tracked: raw scraped data
│   │   ├── yahoo/
│   │   ├── reuters/
│   │   ├── reddit/
│   │   └── twitter/
│   ├── processed/             # DVC-tracked: sentiment-labeled records
│   └── features/              # DVC-tracked: time-series feature CSVs
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── yahoo_finance.py
│   │   ├── reuters_rss.py
│   │   ├── reddit_scraper.py
│   │   ├── twitter_scraper.py
│   │   └── base_scraper.py
│   │
│   ├── sentiment/
│   │   ├── __init__.py
│   │   └── analyzer.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── builder.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── rnn_model.py
│   │   ├── lstm_model.py
│   │   └── gru_model.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       ├── schemas.py
│       └── model_loader.py
│
├── frontend/
│   └── dashboard.py           # Streamlit app
│
├── tests/
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_sentiment.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_training.py
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── Dockerfile.training
│
├── docker-compose.yml          # Local full-stack dev
├── docker-compose.airflow.yml  # Airflow local stack
│
├── dvc.yaml                    # DVC pipeline stages
├── dvc.lock                    # Auto-generated, commit this
├── .dvc/
│   └── config                  # Remote storage config (S3)
│
├── params.yaml                 # All model/training hyperparameters
├── requirements.txt
├── requirements-dev.txt        # pytest, ruff, pre-commit
├── .env.example                # Document all required env vars
├── .gitignore
└── README.md
```

**Rules:**
- `data/` directory is in `.gitignore`. DVC manages it.
- All source code lives in `src/`. No notebooks in the main branch — use a `notebooks/` folder only for exploration and do not include in CI.
- Every module has an `__init__.py`.

---

## 3. Technology Stack

| Layer | Tool | Version Constraint |
|---|---|---|
| Language | Python | 3.11 |
| Deep Learning | PyTorch | >=2.2 |
| Data Manipulation | pandas, numpy | latest stable |
| Sentiment | VADER (nltk) + FinBERT (optional upgrade) | — |
| Yahoo Finance | `yfinance` | >=0.2 |
| Reddit | `praw` | >=7.7 |
| Twitter/X | `snscrape` | latest; fallback if API locked |
| RSS Feeds | `feedparser` | >=6.0 |
| Experiment Tracking | MLflow | >=2.12 |
| Data Versioning | DVC | >=3.49, with S3 remote |
| Pipeline Orchestration | Apache Airflow | 2.9.x (use official Docker image) |
| API | FastAPI + uvicorn | >=0.111 |
| Dashboard | Streamlit | >=1.35 |
| Containerization | Docker + Docker Compose | latest |
| CI/CD | GitHub Actions | — |
| Cloud | AWS EC2 (Ubuntu 22.04 t3.medium minimum) | — |
| Object Storage | AWS S3 (DVC remote + model registry) | — |
| Linting | ruff | >=0.4 |
| Testing | pytest + pytest-cov | >=8.0 |

---

## 4. Environment & Secrets Management

### `.env.example` (commit this file; never commit `.env`)

```
# Reddit
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=market-pulse-predictor/1.0

# Twitter/X (optional — snscrape used if absent)
TWITTER_BEARER_TOKEN=

# Yahoo Finance
# No auth required — yfinance is public

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=market_pulse

# DVC Remote (S3)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1
DVC_REMOTE_BUCKET=market-pulse-dvc

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
AIRFLOW__CORE__FERNET_KEY=          # generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
AIRFLOW__WEBSERVER__SECRET_KEY=     # any random string

# API
API_HOST=0.0.0.0
API_PORT=8000
MODEL_REGISTRY_PATH=./models/       # local fallback; production uses S3

# Tickers to track (comma-separated)
TARGET_TICKERS=AAPL,MSFT,GOOGL,AMZN,TSLA,SPY
```

All application code must read configuration exclusively through environment variables or `params.yaml`. No hardcoded strings for secrets or paths.

---

## Phase 0 — Infrastructure Setup

**Goal:** Runnable skeleton with all services wired together and a passing test suite.

### 0.1 — Repository Init

1. Create the repository with the exact structure from Section 2.
2. Initialize git. Set up `.gitignore` to exclude: `data/`, `*.env`, `__pycache__/`, `.pytest_cache/`, `mlruns/` (MLflow local), `*.pyc`, `models/`.
3. Create `requirements.txt` and `requirements-dev.txt` with all packages from Section 3.
4. Create `.env.example` exactly as shown in Section 4.

### 0.2 — DVC Init

```bash
dvc init
dvc remote add -d s3remote s3://<DVC_REMOTE_BUCKET>/dvc-store
dvc remote modify s3remote region us-east-1
```

Create `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/features/.gitkeep` so empty directories are committed.

Add `data/` to `.dvc/config` as a tracked path. Do not add individual files yet — that happens in Phase 1.

### 0.3 — MLflow Server (local dev)

MLflow runs as a service in `docker-compose.yml`. It stores artifacts to S3 and metadata to a local SQLite DB for development. In production (EC2), it uses a PostgreSQL backend.

`docker-compose.yml` must define these services:
- `mlflow` — image: `ghcr.io/mlflow/mlflow`, port 5000
- `api` — built from `docker/Dockerfile.api`, port 8000
- `dashboard` — built from `docker/Dockerfile.dashboard`, port 8501
- `postgres` — image: `postgres:15`, used by Airflow in the Airflow compose file

### 0.4 — Airflow Stack (separate compose)

Use the official Airflow Docker Compose from Apache (`docker-compose.airflow.yml`). Mount `./airflow/dags` into the container. All DAGs must import from `src/` — add `src/` to `PYTHONPATH` in the Airflow environment.

Services required: `airflow-webserver`, `airflow-scheduler`, `airflow-triggerer`, `postgres`.

### 0.5 — Pre-commit & Linting

Configure `ruff` to lint and format. Add a `.pre-commit-config.yaml` running `ruff check` and `ruff format --check` on staged files. This same check runs in CI.

### 0.6 — Phase 0 Tests

Write these tests in `tests/conftest.py` and `tests/test_api.py` before any real implementation:

- `test_health_endpoint_returns_200`: call `GET /health`, assert status 200 and body `{"status": "ok"}`.
- `test_env_example_has_all_required_keys`: parse `.env.example`, assert the set of keys matches a hardcoded list of required keys.
- `test_docker_compose_valid`: run `docker-compose config --quiet` via subprocess, assert exit code 0.

**All Phase 0 tests must pass before moving to Phase 1.**

---

## Phase 1 — Data Ingestion

**Goal:** Fetch raw data from all four sources. Store as structured JSON/CSV under `data/raw/`. Track with DVC.

### 1.1 — Base Scraper Interface

File: `src/ingestion/base_scraper.py`

Define an abstract base class `BaseScraper` with:
- Abstract method `fetch(ticker: str, lookback_hours: int) -> list[dict]`
- Concrete method `save(records: list[dict], source: str, ticker: str)` — writes a timestamped JSON file to `data/raw/<source>/<ticker>_<timestamp>.json`
- Concrete method `load_latest(source: str, ticker: str) -> list[dict]` — reads the most recent file for a given source and ticker

All records returned by `fetch()` must conform to the **RawRecord schema** (see Section 15).

### 1.2 — Yahoo Finance Scraper

File: `src/ingestion/yahoo_finance.py`

Class `YahooFinanceScraper(BaseScraper)`:

- Use `yfinance.Ticker(ticker).history(period="1d", interval="1h")` to get OHLCV data.
- Also fetch `yfinance.Ticker(ticker).news` for recent headlines.
- Map each price row to a RawRecord with `source="yahoo_price"`, `text=None`, and price fields populated.
- Map each news item to a RawRecord with `source="yahoo_news"`, `text=<headline + summary>`, price fields `None`.
- Supported tickers come from the `TARGET_TICKERS` environment variable.
- No auth required.

### 1.3 — Reuters RSS Scraper

File: `src/ingestion/reuters_rss.py`

Class `ReutersRSSScraper(BaseScraper)`:

- Use `feedparser.parse()` on these feeds (hardcode as class-level constants, all are public):
  - `https://feeds.reuters.com/reuters/businessNews`
  - `https://feeds.reuters.com/reuters/technologyNews`
  - `https://feeds.reuters.com/reuters/marketsNews` (if active)
  - Fallback: Reuters does rotate feed URLs — also support a configurable `REUTERS_RSS_URLS` env var (comma-separated list) to allow overriding without code changes.
- Each entry becomes a RawRecord with `source="reuters"`, `text=<title + summary>`, ticker matched by keyword scan against `TARGET_TICKERS` (or `ticker="MARKET"` if general).
- Filter entries to only those published within `lookback_hours`.
- Deduplicate by `entry.id` or link hash. Store dedup state in a small JSON sidecar file `data/raw/reuters/seen_ids.json`.

### 1.4 — Reddit Scraper

File: `src/ingestion/reddit_scraper.py`

Class `RedditScraper(BaseScraper)`:

- Use `praw.Reddit` authenticated with `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`.
- Subreddits to monitor: `r/stocks`, `r/investing`, `r/wallstreetbets`, `r/finance`, `r/StockMarket`.
- Fetch top 50 `hot` posts and their top 10 comments per subreddit per run.
- Map each post to a RawRecord: `source="reddit"`, `text=<title + selftext>`, `score=<upvote_ratio>`.
- Map each comment similarly with `source="reddit_comment"`.
- Deduplicate by `post.id` / `comment.id` using the same sidecar JSON pattern as Reuters.
- Ticker matching: scan text for mentions of any ticker in `TARGET_TICKERS` using a regex word-boundary match. A record may map to multiple tickers — emit one RawRecord per ticker matched, or `ticker="MARKET"` if none found.

### 1.5 — Twitter/X Scraper

File: `src/ingestion/twitter_scraper.py`

Class `TwitterScraper(BaseScraper)`:

**Primary strategy — Bearer Token API (v2):**  
If `TWITTER_BEARER_TOKEN` is set, use `httpx` to call `GET https://api.twitter.com/2/tweets/search/recent` with query `($TICKER OR #$TICKER) lang:en -is:retweet` for each ticker. Max 100 results per request. Respect rate limits with exponential backoff.

**Fallback strategy — snscrape:**  
If the bearer token is not set (or the API returns 429/403), fall back to `snscrape.modules.twitter.TwitterSearchScraper`. Query: `"$TICKER" OR "#$TICKER" lang:en`. Limit to 100 tweets per ticker per run. snscrape works without auth.

Both paths produce RawRecords with `source="twitter"`.

This dual-path design means the system works even with no API credentials.

### 1.6 — DVC Stage: Raw Data

After all scrapers exist, define a DVC stage in `dvc.yaml`:

```yaml
stages:
  ingest:
    cmd: python -m src.ingestion.run_all --lookback-hours 24
    deps:
      - src/ingestion/
    outs:
      - data/raw/
    params:
      - params.yaml:
          - ingestion.target_tickers
          - ingestion.lookback_hours
```

Create `src/ingestion/run_all.py` as the entry point that instantiates all four scrapers and runs them sequentially.

### 1.7 — Phase 1 Tests

Write tests **before** implementing the scrapers:

- `test_yahoo_finance_fetch_returns_list`: mock `yfinance.Ticker`, assert output is a list of dicts matching RawRecord schema.
- `test_reuters_rss_deduplication`: feed the scraper two identical entries, assert only one RawRecord is stored.
- `test_reddit_scraper_ticker_matching`: given a post with text "I'm bullish on AAPL today", assert output contains `ticker="AAPL"`.
- `test_twitter_scraper_falls_back_to_snscrape`: with no `TWITTER_BEARER_TOKEN` env var, assert the fallback path is called.
- `test_base_scraper_save_creates_file`: call `save()`, assert file exists at expected path with correct JSON structure.
- `test_run_all_produces_raw_files`: integration test — run `run_all` with mocked scrapers, assert files are created under `data/raw/`.

---

## Phase 2 — Sentiment Analysis

**Goal:** Classify each text record as Positive, Negative, or Neutral. Store labeled records in `data/processed/`.

### 2.1 — Analyzer

File: `src/sentiment/analyzer.py`

Class `SentimentAnalyzer`:

**Primary method: VADER**  
Use `nltk.sentiment.vader.SentimentIntensityAnalyzer`. VADER's compound score maps to labels as:
- compound >= 0.05 → `POSITIVE`
- compound <= -0.05 → `NEGATIVE`
- otherwise → `NEUTRAL`

**Why VADER:** It is rule-based, needs no GPU, no API calls, zero latency, and was specifically designed for social media short text — a strong fit for tweets and Reddit posts.

**Optional FinBERT path (upgrade):**  
If the environment variable `USE_FINBERT=1` is set, load `ProsusAI/finbert` via the HuggingFace `transformers` pipeline for finance-domain accuracy. This path is only used if a GPU is available; otherwise fall back to VADER automatically.

Public method signature:
```python
def analyze(self, text: str) -> dict:
    # returns: {"label": "POSITIVE"|"NEGATIVE"|"NEUTRAL", "score": float}
```

```python
def batch_analyze(self, records: list[dict]) -> list[dict]:
    # Adds "sentiment_label" and "sentiment_score" to each record.
    # Skips records where text is None (price records).
    # Returns the enriched list.
```

### 2.2 — Processing Entry Point

File: `src/sentiment/run_sentiment.py`

- Load all files from `data/raw/` that have not yet been processed (track processed filenames in `data/processed/processed_files.json`).
- Run `batch_analyze()` on each batch.
- Write output to `data/processed/<source>/<ticker>_<timestamp>.parquet` using pandas. Use Parquet for efficient storage and fast reads downstream.

### 2.3 — DVC Stage: Sentiment

```yaml
  sentiment:
    cmd: python -m src.sentiment.run_sentiment
    deps:
      - src/sentiment/
      - data/raw/
    outs:
      - data/processed/
    params:
      - params.yaml:
          - sentiment.method   # "vader" or "finbert"
```

### 2.4 — Phase 2 Tests

- `test_vader_positive_text`: pass "Stock market rally! Strong earnings beat expectations." — assert label is `POSITIVE`.
- `test_vader_negative_text`: pass "Market crash fears grow as recession looms." — assert label is `NEGATIVE`.
- `test_vader_neutral_text`: pass "The market opened today." — assert label is `NEUTRAL`.
- `test_batch_analyze_skips_none_text`: include a record with `text=None`, assert it passes through without error and `sentiment_label` is `None`.
- `test_batch_analyze_returns_enriched_records`: assert all text records have `sentiment_label` and `sentiment_score` keys added.
- `test_run_sentiment_writes_parquet`: mock analyzer, assert output `.parquet` files are created.

---

## Phase 3 — Time-Series Feature Construction

**Goal:** Aggregate sentiment and price data into a structured time-series dataset suitable for RNN/LSTM/GRU input.

### 3.1 — Feature Builder

File: `src/features/builder.py`

Class `TimeSeriesBuilder`:

Input: All processed Parquet files from `data/processed/`.

**Step 1 — Aggregate sentiment per ticker per hour:**

Group records by `(ticker, hour_bucket)`. For each bucket compute:
- `sentiment_positive_count`: int
- `sentiment_negative_count`: int
- `sentiment_neutral_count`: int
- `sentiment_score_mean`: float (mean of sentiment scores)
- `sentiment_score_std`: float
- `total_mentions`: int
- `reddit_mentions`, `twitter_mentions`, `news_mentions`: int (by source)

**Step 2 — Merge with Yahoo Finance price data:**

Join on `(ticker, hour_bucket)`. Price columns to include:
- `open`, `high`, `low`, `close`, `volume`
- `price_return`: `(close - prev_close) / prev_close`
- `price_return_5h`: 5-hour rolling return
- `volatility_1h`: rolling 1-hour standard deviation of `price_return`
- `volatility_6h`: rolling 6-hour standard deviation

**Step 3 — Construct labels:**

From the price data, create three target columns:
- `label_direction`: `1` if next-hour `price_return` > 0.001, `-1` if < -0.001, else `0` (Up/Down/Flat)
- `label_return`: next-hour `price_return` (regression target)
- `label_volatility_spike`: `1` if next-hour `volatility_1h` > 2 standard deviations above rolling mean, else `0`

**Step 4 — Sequence creation:**

Create sliding windows of `SEQUENCE_LENGTH` hours (default 24, configurable in `params.yaml`). Each sample is an array of shape `(SEQUENCE_LENGTH, NUM_FEATURES)`. Features are all numeric columns except the three label columns. Exclude the ticker column.

**Step 5 — Train/Val/Test split:**

Split chronologically (not randomly) to prevent data leakage:
- Train: earliest 70% of time
- Validation: next 15%
- Test: most recent 15%

Save to `data/features/`:
- `X_train.npy`, `y_direction_train.npy`, `y_return_train.npy`, `y_volatility_train.npy`
- Same for `val` and `test`
- `feature_columns.json` — ordered list of feature names (critical for consistent inference)
- `scaler.pkl` — fitted `sklearn.preprocessing.StandardScaler` (fit only on train, transform all splits)

### 3.2 — DVC Stage: Features

```yaml
  features:
    cmd: python -m src.features.run_features
    deps:
      - src/features/
      - data/processed/
    outs:
      - data/features/
    params:
      - params.yaml:
          - features.sequence_length
          - features.train_ratio
          - features.val_ratio
```

### 3.3 — Phase 3 Tests

- `test_sentiment_aggregation_correct_counts`: given 3 positive and 2 negative records in the same hour bucket, assert output counts match.
- `test_feature_merge_no_price_no_row`: if a time bucket has sentiment but no price data, assert that row is dropped.
- `test_label_direction_correct`: given a price return of +0.005 for the next hour, assert `label_direction == 1`.
- `test_volatility_spike_detection`: construct a price series with a known spike, assert the spike bucket is labeled `1`.
- `test_chronological_split_no_leakage`: assert max timestamp in train < min timestamp in validation < min timestamp in test.
- `test_scaler_fit_on_train_only`: assert the scaler's mean values are derived only from train data.
- `test_sequence_shape`: assert `X_train.shape[1]` equals `SEQUENCE_LENGTH` and `X_train.shape[2]` equals the number of feature columns.

---

## Phase 4 — Model Training & Experiment Tracking

**Goal:** Train three models (SimpleRNN, LSTM, GRU) with MLflow tracking. Register the best model.

### 4.1 — `params.yaml`

This file is the single source of truth for all hyperparameters. Structure:

```yaml
ingestion:
  target_tickers: "AAPL,MSFT,GOOGL,AMZN,TSLA,SPY"
  lookback_hours: 24

sentiment:
  method: "vader"

features:
  sequence_length: 24
  train_ratio: 0.70
  val_ratio: 0.15

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  early_stopping_patience: 7
  task: "direction"    # "direction" | "return" | "volatility"
  model: "lstm"        # "rnn" | "lstm" | "gru"
  device: "auto"       # "auto" | "cpu" | "cuda"
```

### 4.2 — Base Model

File: `src/models/base_model.py`

Abstract `torch.nn.Module` subclass `BaseSequentialModel`:
- Constructor accepts: `input_size`, `hidden_size`, `num_layers`, `output_size`, `dropout`.
- Abstract method `forward(x: Tensor) -> Tensor`.
- Concrete method `count_parameters() -> int`.

### 4.3 — Three Model Implementations

**`src/models/rnn_model.py` — `SimpleRNNModel(BaseSequentialModel)`:**
- Uses `torch.nn.RNN`.
- Takes the last hidden state and passes through a linear layer.
- Output size: 3 for direction (softmax), 1 for return (linear), 2 for volatility (softmax).

**`src/models/lstm_model.py` — `LSTMModel(BaseSequentialModel)`:**
- Uses `torch.nn.LSTM`.
- Takes the last hidden state (from `h_n`, not the cell state).
- Same output structure as RNN.

**`src/models/gru_model.py` — `GRUModel(BaseSequentialModel)`:**
- Uses `torch.nn.GRU`.
- Takes the last hidden state.
- Same output structure.

All three must have identical constructor signatures. The trainer instantiates whichever is named in `params.yaml`.

### 4.4 — Trainer

File: `src/training/trainer.py`

Class `Trainer`:

Constructor accepts: model, train dataloader, val dataloader, learning rate, device, patience, task name.

The `train()` method must:
1. Use `Adam` optimizer.
2. Loss function chosen by task:
   - `direction` → `CrossEntropyLoss`
   - `return` → `MSELoss`
   - `volatility` → `BCEWithLogitsLoss`
3. Run for up to `epochs` from `params.yaml`.
4. Implement early stopping on validation loss (stop if no improvement for `patience` epochs).
5. Save the best model checkpoint to `models/<model_name>_<task>_best.pt`.
6. Log to MLflow every epoch:
   - `train_loss`, `val_loss`, `val_accuracy` (for classification tasks), `val_f1`, `val_rmse` (for regression).
7. Log all hyperparameters from `params.yaml` at run start via `mlflow.log_params()`.
8. Log the final model with `mlflow.pytorch.log_model()`.

### 4.5 — Evaluator

File: `src/training/evaluator.py`

Class `Evaluator`:

Takes a trained model and test dataloader. Computes and returns:
- For classification: `accuracy`, `f1_score` (macro), `confusion_matrix`, `classification_report`.
- For regression: `rmse`, `mae`.

All metrics are returned as a dict and logged to MLflow.

### 4.6 — Training Entry Point

File: `src/training/run_training.py`

CLI script that:
1. Loads `params.yaml`.
2. Loads `data/features/` numpy arrays.
3. Creates DataLoaders.
4. Instantiates the model specified in `params.yaml`.
5. Starts an MLflow run under experiment `MLFLOW_EXPERIMENT_NAME`.
6. Calls `Trainer.train()` then `Evaluator.evaluate()`.
7. Uses `mlflow.register_model()` to register the model as `MarketPulsePredictor-<task>`. If this is the best-performing run for this task (highest accuracy / lowest RMSE), promote it to `Production` stage.

**To train all three models for all three tasks, run the script three times with different `params.yaml` `model` values.** Automate this in the Airflow training DAG.

### 4.7 — DVC Stage: Training

```yaml
  train:
    cmd: python -m src.training.run_training
    deps:
      - src/models/
      - src/training/
      - data/features/
    outs:
      - models/
    params:
      - params.yaml:
          - training
```

### 4.8 — Phase 4 Tests

- `test_rnn_forward_pass_shape`: instantiate `SimpleRNNModel`, pass a batch of `(32, 24, 10)`, assert output shape is `(32, 3)` for direction task.
- `test_lstm_forward_pass_shape`: same as above for LSTM.
- `test_gru_forward_pass_shape`: same as above for GRU.
- `test_trainer_runs_one_epoch`: mock dataloader with 2 batches, call `train()` with `epochs=1`, assert no error and checkpoint file created.
- `test_early_stopping_triggers`: mock validation loss to not improve for `patience+1` epochs, assert training stops before max epochs.
- `test_evaluator_returns_all_metrics`: pass mock predictions and targets, assert output dict contains `accuracy`, `f1_score`, `confusion_matrix`.
- `test_mlflow_run_logged`: after a mock training run, assert MLflow experiment contains at least one run with logged params and metrics.

---

## Phase 5 — REST API

**Goal:** Expose predictions and experiment results over HTTP.

### 5.1 — Schemas

File: `src/api/schemas.py`

Define Pydantic v2 models:

```python
class PredictionRequest:
    ticker: str
    # If provided, use this pre-built feature vector. Otherwise, build from live data.
    feature_override: list[float] | None = None

class PredictionResponse:
    ticker: str
    direction: str          # "UP" | "DOWN" | "FLAT"
    direction_confidence: float
    predicted_return: float
    volatility_spike: bool
    volatility_confidence: float
    model_name: str
    timestamp: str          # ISO 8601

class ModelComparisonResponse:
    models: list[dict]      # Each dict: model_name, task, accuracy, f1, rmse, run_id

class HealthResponse:
    status: str
    model_loaded: bool
    last_data_ingestion: str | None
```

### 5.2 — Model Loader

File: `src/api/model_loader.py`

Class `ModelRegistry`:
- On startup, load the `Production` stage model for each task from the MLflow model registry.
- If MLflow is unreachable, fall back to loading the latest `.pt` checkpoint from the `models/` directory.
- Cache loaded models in memory (one per task).
- Expose `get_model(task: str)` and `get_latest_features(ticker: str)` (reads most recent features from `data/features/`).

### 5.3 — API Routes

File: `src/api/main.py`

FastAPI app. All routes prefixed with `/api/v1`.

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/predict` | Run prediction for a ticker |
| GET | `/results` | Return latest model comparison metrics from MLflow |
| GET | `/experiments` | List all MLflow runs with key metrics |
| GET | `/tickers` | Return the list of tracked tickers |
| GET | `/sentiment/{ticker}` | Return last 24h of aggregated sentiment for a ticker |
| GET | `/prices/{ticker}` | Return last 24h of OHLCV for a ticker |

**`POST /predict` logic:**
1. Load the most recent 24-hour feature window for the requested ticker from `data/features/`.
2. Run all three production models (direction, return, volatility).
3. Return a `PredictionResponse`.

All routes must return proper HTTP error codes: 404 if ticker not tracked, 503 if models not loaded, 422 for bad input.

Add CORS middleware to allow all origins (needed for the dashboard).

### 5.4 — Phase 5 Tests

- `test_health_endpoint_returns_200`: GET `/health`, assert 200 and `status == "ok"`.
- `test_predict_returns_valid_schema`: POST `/predict` with mocked model loader, assert response matches `PredictionResponse` schema.
- `test_predict_unknown_ticker_returns_404`: POST with `ticker="FAKE"`, assert 404.
- `test_results_endpoint_returns_list`: GET `/results`, assert response is a list with at least the expected keys.
- `test_cors_header_present`: assert `Access-Control-Allow-Origin` header is in response.

Use `httpx.AsyncClient` with the FastAPI `app` directly (no running server needed in tests).

---

## Phase 6 — Frontend Dashboard

**Goal:** A single Streamlit app that gives a complete view of the system's state and predictions.

File: `frontend/dashboard.py`

### 6.1 — Pages / Sections

The dashboard is a single-page app with a sidebar for navigation between these sections:

**Section 1 — Live Predictions**
- Dropdown to select a ticker from the tracked list (fetched from `GET /tickers`).
- "Predict Now" button that calls `POST /predict`.
- Display three result cards side by side:
  - Card 1: Market Direction (UP/DOWN/FLAT with confidence bar)
  - Card 2: Predicted Return (percentage with color coding: green positive, red negative)
  - Card 3: Volatility Spike (YES/NO with confidence)
- Auto-refresh every 5 minutes using `st.rerun()` with a countdown timer.

**Section 2 — Sentiment Feed**
- Select a ticker.
- Fetch `GET /sentiment/{ticker}`.
- Display a bar chart (Plotly) of sentiment distribution (positive/negative/neutral) over the last 24 hours, grouped by hour.
- Display a data table of the raw sentiment records below the chart.

**Section 3 — Price Chart**
- Select a ticker.
- Fetch `GET /prices/{ticker}`.
- Display an interactive Plotly candlestick chart of the last 24 hours.
- Overlay sentiment score as a line on a secondary y-axis.

**Section 4 — Model Comparison**
- Fetch `GET /results`.
- Display a table comparing all three models (RNN, LSTM, GRU) across all three tasks.
- Columns: Model, Task, Accuracy, F1 Score, RMSE, Training Duration, MLflow Run ID.
- Highlight the best model per task in green.
- Display training vs. validation loss curves for each model (fetched from MLflow via `mlflow.get_metric_history()`).

**Section 5 — Pipeline Status**
- Display the status of the last Airflow DAG runs by polling the Airflow REST API (`GET /api/v1/dags/{dag_id}/dagRuns`).
- Show: DAG name, last run time, duration, status (success/failed/running).
- Display a simple text log of the most recent DVC pipeline run.

### 6.2 — Dashboard Implementation Notes

- Use `st.cache_data(ttl=60)` on all API calls to avoid hammering the backend on every interaction.
- Handle API errors gracefully: if the API is down, display a warning message instead of crashing.
- The `API_BASE_URL` must be configurable via environment variable `API_BASE_URL` (default: `http://localhost:8000/api/v1`).
- All charts must use Plotly (not Matplotlib) for interactivity.

---

## Phase 7 — Airflow Orchestration

**Goal:** Four DAGs that run the pipeline end-to-end on a schedule.

### 7.1 — DAG Conventions

All DAGs must:
- Set `catchup=False`.
- Define `default_args` with `retries=2`, `retry_delay=timedelta(minutes=5)`, `email_on_failure=False`.
- Use `PythonOperator` or `BashOperator` (no external providers needed).
- Import all business logic from `src/` — DAGs are thin wrappers only.

### 7.2 — DAG Definitions

**`ingestion_dag.py`**
- Schedule: `*/30 * * * *` (every 30 minutes)
- Tasks (in order, each a `PythonOperator`):
  1. `ingest_yahoo` — calls `YahooFinanceScraper().fetch()`
  2. `ingest_reuters` — calls `ReutersRSSScraper().fetch()`
  3. `ingest_reddit` — calls `RedditScraper().fetch()`
  4. `ingest_twitter` — calls `TwitterScraper().fetch()`
  5. `dvc_add_raw` — runs `dvc add data/raw/` via subprocess and commits the `.dvc` lock file to git.
- Tasks 1-4 run in parallel. Task 5 runs after all four complete.

**`sentiment_dag.py`**
- Schedule: `5 * * * *` (5 minutes past every hour — after ingestion completes)
- Tasks:
  1. `run_sentiment` — calls `run_sentiment.py`
  2. `dvc_add_processed` — runs `dvc add data/processed/`

**`feature_dag.py`**
- Schedule: `15 * * * *` (15 minutes past every hour)
- Tasks:
  1. `build_features` — calls `run_features.py`
  2. `dvc_add_features` — runs `dvc add data/features/`

**`training_dag.py`**
- Schedule: `0 2 * * *` (daily at 2 AM — retrain overnight on fresh data)
- Tasks:
  1. `train_rnn` — runs training with `model=rnn`
  2. `train_lstm` — runs training with `model=lstm`
  3. `train_gru` — runs training with `model=gru`
  4. `evaluate_and_register` — evaluator runs on test set for all three, best model promoted to Production in MLflow registry.
- Tasks 1-3 run in parallel. Task 4 runs after all three complete.

---

## Phase 8 — CI/CD with GitHub Actions

### 8.1 — `ci.yml` (runs on every push and PR)

Triggers: `push` to any branch, `pull_request` targeting `main`.

Jobs:

**`lint`:**
- Runner: `ubuntu-latest`
- Steps: checkout, setup Python 3.11, install `requirements-dev.txt`, run `ruff check src/ tests/`, run `ruff format --check src/ tests/`.

**`test`:**
- Runner: `ubuntu-latest`
- Steps: checkout, setup Python 3.11, install `requirements.txt` and `requirements-dev.txt`, run `pytest tests/ --cov=src --cov-report=xml -v`.
- Upload coverage report as artifact.
- Fail the job if coverage drops below 70%.

Both jobs must pass before a PR can be merged (enforce via branch protection rules on `main`).

### 8.2 — `cd.yml` (runs on push to `main` only)

Triggers: `push` to `main`.

**`build-and-push`:**
- Build `docker/Dockerfile.api` and tag as `market-pulse-api:latest` and `market-pulse-api:<sha>`.
- Build `docker/Dockerfile.dashboard` and tag as `market-pulse-dashboard:latest`.
- Push both images to AWS ECR (use `aws-actions/amazon-ecr-login`).
- Secrets required in GitHub Actions: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY`.

**`deploy`:**
- Depends on `build-and-push`.
- SSH into the EC2 instance using a stored private key secret `EC2_SSH_KEY`.
- On the EC2 instance, run:
  ```bash
  docker pull <ECR_URI>/market-pulse-api:latest
  docker pull <ECR_URI>/market-pulse-dashboard:latest
  docker-compose -f /app/docker-compose.yml up -d --no-deps api dashboard
  ```
- This performs a zero-downtime rolling update of only the changed services.

---

## Phase 9 — Docker & AWS EC2 Deployment

### 9.1 — Dockerfiles

**`docker/Dockerfile.api`:**
- Base: `python:3.11-slim`
- Install `requirements.txt`.
- Copy `src/` into `/app/src/`.
- Copy `models/` into `/app/models/` (or pull from S3 at startup — see Model Loader in Phase 5).
- Expose port 8000.
- Command: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2`

**`docker/Dockerfile.dashboard`:**
- Base: `python:3.11-slim`
- Install `requirements.txt` (or a slimmer `requirements-dashboard.txt` with only Streamlit, Plotly, httpx, pandas).
- Copy `frontend/dashboard.py` into `/app/`.
- Expose port 8501.
- Command: `streamlit run /app/dashboard.py --server.port 8501 --server.address 0.0.0.0`

**`docker/Dockerfile.training`:**
- Base: `python:3.11-slim` (or `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime` if GPU EC2 is used)
- Used only for manual re-training runs, not kept running.

### 9.2 — `docker-compose.yml` (production, runs on EC2)

Services:
- `api`: built from `Dockerfile.api`, port `8000:8000`, env from `.env`, restart `unless-stopped`.
- `dashboard`: built from `Dockerfile.dashboard`, port `8501:8501`, env from `.env`, restart `unless-stopped`.
- `mlflow`: `ghcr.io/mlflow/mlflow`, port `5000:5000`, data volume mounted to `/mlflow`, restart `unless-stopped`. In production, use `--backend-store-uri postgresql://...` and `--default-artifact-root s3://...`.
- `nginx` (optional but recommended): reverse proxy to expose port 80 → API on 8000 and port 81 → dashboard on 8501. Use `nginx:alpine`. Include a simple `nginx.conf`.

All services share a Docker network `market-pulse-net`.

### 9.3 — EC2 Setup (one-time, manual)

Document these steps in `README.md` under "EC2 Deployment":

1. Launch `t3.medium` EC2 instance with Ubuntu 22.04.
2. Open inbound ports: 22 (SSH), 80, 8000, 8501, 5000 (restrict 5000 to your IP only).
3. Install Docker and Docker Compose on the instance.
4. Clone the repository to `/app/`.
5. Copy `.env` to `/app/.env` (copy manually, never commit).
6. Configure AWS CLI with credentials that have S3 access for DVC.
7. Run `dvc pull` to fetch the latest tracked data.
8. Run `docker-compose up -d` to start all services.
9. Set up a cron job or systemd service to run the Airflow Docker stack on startup.

---

## Data Schema Reference

### RawRecord (all ingestion outputs)

```json
{
  "id": "unique hash or platform ID",
  "source": "yahoo_price | yahoo_news | reuters | reddit | reddit_comment | twitter",
  "ticker": "AAPL",
  "timestamp": "2024-01-15T14:00:00Z",
  "text": "string or null",
  "open": 182.50,
  "high": 183.20,
  "low": 182.10,
  "close": 183.00,
  "volume": 45000000,
  "score": 0.85,
  "url": "https://..."
}
```

Price fields (`open`, `high`, `low`, `close`, `volume`) are `null` for text-only records. `text` is `null` for price-only records. `score` is Reddit upvote ratio or Twitter engagement metric (null for news/price).

### ProcessedRecord (output of sentiment analysis)

Extends RawRecord with:
```json
{
  "sentiment_label": "POSITIVE | NEGATIVE | NEUTRAL | null",
  "sentiment_score": 0.72
}
```

### Feature Row (one row per ticker per hour)

All columns from sentiment aggregation + price data + computed labels, as described in Phase 3. The exact column list is persisted in `data/features/feature_columns.json` and must be used consistently at inference time.

---

## Testing Strategy

### Principles

- Write tests **before** implementation in each phase (TDD).
- Tests must not make real network calls. Mock all external APIs using `unittest.mock.patch` or `pytest-mock`.
- Tests must not read or write to real `data/` directories. Use `tmp_path` fixtures from pytest.
- No test should depend on another test's output.

### Test Categories

**Unit tests** (`tests/test_*.py`): test one function or class in isolation. All dependencies mocked.

**Integration tests** (tagged with `@pytest.mark.integration`): test two or more real modules together. Do not mock internal modules, but still mock external APIs. Run locally but skipped in CI (too slow).

**API tests** (`tests/test_api.py`): use `httpx.AsyncClient(app=app, base_url="http://test")` to test all endpoints without a running server. Model loading mocked.

### Coverage Target

- Overall: >= 70%
- `src/models/`: >= 85%
- `src/api/`: >= 90%
- `src/sentiment/`: >= 80%

---

## MLflow Experiment Convention

- Experiment name: `market_pulse` (set via `MLFLOW_EXPERIMENT_NAME` env var)
- Run naming: `<model_name>_<task>_<YYYYMMDD_HHMM>` (e.g., `lstm_direction_20240115_0200`)
- Tags to set on every run:
  - `model_type`: `rnn | lstm | gru`
  - `task`: `direction | return | volatility`
  - `data_version`: output of `dvc data status --json` hash
- Registered model names:
  - `MarketPulsePredictor-direction`
  - `MarketPulsePredictor-return`
  - `MarketPulsePredictor-volatility`
- Stages: `Staging` → `Production`. Only one model per task lives in `Production` at any time.

---

## DVC Data Versioning Convention

- Remote name: `s3remote`
- Remote URL: `s3://<DVC_REMOTE_BUCKET>/dvc-store`
- Tracked paths: `data/raw/`, `data/processed/`, `data/features/`, `models/`
- Every Airflow DAG that writes to a tracked path must call `dvc add <path>` afterward. The resulting `.dvc` file is committed to git by the DAG using a git commit via subprocess.
- Use `dvc repro` to replay the full DVC pipeline from scratch (ingestion → sentiment → features → training). This must work end-to-end.

---

## Known Constraints & Mitigations

| Constraint | Mitigation |
|---|---|
| Twitter/X API free tier is severely rate-limited (1 app/15-min window) | Use snscrape as the primary method; store the bearer token only for supplemental enrichment |
| Reuters RSS feed URLs change periodically | Support `REUTERS_RSS_URLS` env var override; document current working URLs in README |
| FinBERT requires GPU for reasonable throughput | Default to VADER; enable FinBERT only via `USE_FINBERT=1` env var and only when CUDA is detected |
| Airflow git-auto-commit from inside Docker requires SSH key setup | Mount an SSH agent socket into the Airflow container; document this in the README EC2 section |
| Cold start: no data on first run | Provide a `scripts/backfill.py` script that runs ingestion with `--lookback-hours 168` (7 days) using yfinance historical data to bootstrap the dataset |
| EC2 t3.medium may be tight on RAM for training | Training DAG runs in a separate Docker container with `--memory 3g` limit; use `batch_size=32` if OOM errors occur |
| Reddit PRAW requires app registration | Document step-by-step Reddit app creation in README; this is free |

---

*End of specification. Implement phase by phase. Do not skip phases or merge them. Run tests at the end of each phase before proceeding.*
