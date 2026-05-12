# market-pulse-predictor

Real-time market movement prediction pipeline covering ingestion, sentiment analysis,
time-series feature generation, and sequential model training.

Notes:
- Yahoo Finance ingestion is intentionally rate-limited and cached to reduce breakage risk.
- Reddit uses script auth and now expects username/password in addition to app credentials.
- StockTwits is the unauthenticated social-text source.
- Alpha Vantage is the primary structured news API and can provide pre-computed sentiment.

