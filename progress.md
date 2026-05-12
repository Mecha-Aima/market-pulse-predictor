# Progress

## Current state

Implemented through spec Phase 4, with the ingestion pipeline updated for the May 2026 corrections.

## Completed

- Phase 0 skeleton:
  - Repo structure
  - Compose files
  - DVC stage file
  - Ruff and pre-commit setup
  - Minimal API health endpoint

- Phase 1 ingestion:
  - `BaseScraper` save/load helpers and dedup sidecar helpers
  - `YahooFinanceScraper`
    - Per-ticker fetch flow
    - Cached/rate-limited session builder
    - `YFRateLimitError` retry path with backoff and skip-on-repeat
  - `NewsRSSScraper`
    - Replaces old Reuters-specific module
    - Uses `NEWS_RSS_URLS` override
    - Writes to `data/raw/news_rss/`
  - `RedditScraper`
    - Uses script auth fields including username/password
    - Reduced fetch limits to 25 posts and 5 comments
  - `StockTwitsScraper`
    - Replaces Twitter/X completely
    - Uses public unauthenticated symbol stream endpoint
    - Deduplicates by message id
  - `AlphaVantageNewsScraper`
    - Primary news API source
    - Batches configured tickers in one request
    - Preserves Alpha Vantage sentiment fields on raw records
  - `run_all.py`
    - Updated scraper registry to Yahoo + RSS + Reddit + StockTwits + Alpha Vantage

- Phase 2 sentiment:
  - `SentimentAnalyzer.batch_analyze()` now short-circuits VADER when
    `av_sentiment_label` is present on a record
  - Mapping:
    - `Bullish -> POSITIVE`
    - `Bearish -> NEGATIVE`
    - `Neutral -> NEUTRAL`

- Phase 3 features:
  - Kept the output schema unchanged
  - Mapped `stocktwits` into the existing `twitter_mentions` feature bucket for compatibility
  - Expanded `news_mentions` to include `news_rss`, `yahoo_news`, and `alphavantage_news`

- Phase 4 training:
  - No model/training architecture changes required for this ingestion correction

## Files changed for the May 2026 correction

- Added:
  - `src/ingestion/news_rss.py`
  - `src/ingestion/stocktwits_scraper.py`
  - `src/ingestion/alphavantage_scraper.py`
  - `progress.md`

- Removed:
  - `src/ingestion/reuters_rss.py`
  - `src/ingestion/twitter_scraper.py`

- Updated:
  - `.env.example`
  - `requirements.txt`
  - `README.md`
  - `src/ingestion/base_scraper.py`
  - `src/ingestion/yahoo_finance.py`
  - `src/ingestion/reddit_scraper.py`
  - `src/ingestion/run_all.py`
  - `src/sentiment/analyzer.py`
  - `src/features/builder.py`
  - `tests/conftest.py`
  - `tests/test_ingestion.py`
  - `tests/test_sentiment.py`
  - `tests/test_features.py`

## Remaining work

- Phase 5 and beyond from the original project spec are still not implemented:
  - REST API completion
  - Model registry-backed loader behavior
  - Dashboard
  - Airflow DAG implementations
  - CI/CD workflows
  - Deployment polish

- The placeholder workflow and DAG files are still placeholders.

- Real dependency installation has not been performed in this handoff.
  - The code was kept import-safe where possible so the repo can still be edited/tested in partial environments.

## Important pointers for the next agent

- The user later explicitly said not to run tests in the latest instruction, so no further verification should be done unless they re-approve it.

- `YahooFinanceScraper._build_session()` currently tries to honor the requested cache and rate-limit pattern while remaining resilient if the packages are not installed yet.
  - If runtime validation is needed later, install dependencies first and check the concrete behavior of the combined session object with the installed library versions.

- Alpha Vantage free-tier limits are tight.
  - Keep ticker batching in place.
  - Avoid changing it back to one request per ticker.

- The feature schema was intentionally left backward-compatible.
  - `twitter_mentions` is now effectively the StockTwits bucket.
  - Changing the feature column names would cascade into saved artifacts, training, and inference.

- Raw records may now contain extra optional keys like:
  - `av_sentiment_label`
  - `av_sentiment_score`
  - `BaseScraper` now preserves unknown extra keys instead of dropping them.

- If a later agent touches ingestion again, re-check:
  - Yahoo Finance release notes
  - Alpha Vantage docs for `NEWS_SENTIMENT`
  - PRAW script-auth docs

## Suggested next action

If the next agent is continuing beyond this correction, the best next task is Phase 5 API implementation, starting with the model loader and prediction routes, while keeping the updated feature schema stable.
