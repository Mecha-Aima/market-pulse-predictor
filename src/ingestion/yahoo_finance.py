from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from time import sleep
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

try:
    import requests_cache
except ImportError:  # pragma: no cover
    requests_cache = None

try:
    from requests_ratelimiter import LimiterSession
except ImportError:  # pragma: no cover
    LimiterSession = None

from src.ingestion.base_scraper import BaseScraper

LOGGER = logging.getLogger(__name__)


class YahooFinanceScraper(BaseScraper):
    source_name = "yahoo"

    def __init__(self) -> None:
        pass  # yfinance 1.3+ manages its own curl_cffi session; don't inject one

    def fetch(self, ticker: str, lookback_hours: int) -> list[dict]:
        if yf is None or pd is None:
            raise RuntimeError("yfinance and pandas are required for Yahoo ingestion")

        period, interval = self._period_interval(lookback_hours)
        client = self._build_ticker(ticker)
        history = self._call_with_retry(
            lambda: client.history(period=period, interval=interval), ticker
        )
        news_items = self._call_with_retry(lambda: client.news or [], ticker)
        if history is None:
            return []

        records = self._build_price_records(ticker, history)
        records.extend(self._build_news_records(ticker, news_items, lookback_hours))
        return records

    @staticmethod
    def _period_interval(lookback_hours: int) -> tuple[str, str]:
        """Map lookback hours to the best yfinance period+interval combination.

        Always uses daily ('1d') bars to match the daily feature pipeline.
        """
        if lookback_hours <= 720:    # <= 30 days
            return "1mo", "1d"
        if lookback_hours <= 2160:   # <= 90 days
            return "3mo", "1d"
        if lookback_hours <= 8760:   # <= 1 year
            return "1y", "1d"
        if lookback_hours <= 17520:  # <= 2 years
            return "2y", "1d"
        return "5y", "1d"           # max free via yfinance

    def _build_session(self):
        cache_dir = self._raw_dir(self.source_name)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_session = None
        if requests_cache is not None:
            cache_session = requests_cache.CachedSession(
                str(cache_dir / ".yf_cache"),
                expire_after=1800,
            )

        if LimiterSession is None:
            return cache_session

        limiter_session = LimiterSession(per_second=0.5)
        if cache_session is not None:
            limiter_session.cookies = cache_session.cookies
            limiter_session.cache = getattr(cache_session, "cache", None)
        return limiter_session

    def _build_ticker(self, ticker: str):
        # Do NOT pass a session — yfinance 1.3+ uses curl_cffi internally
        # and raises YFDataException if any custom session is injected.
        return yf.Ticker(ticker)

    def _call_with_retry(self, operation, ticker: str):
        rate_limit_error = self._rate_limit_error()
        try:
            return operation()
        except rate_limit_error:
            sleep(90)
            try:
                return operation()
            except rate_limit_error:
                LOGGER.warning("Yahoo rate limit persisted for %s; skipping ticker", ticker)
                return None

    def _rate_limit_error(self):
        exceptions = getattr(yf, "exceptions", None)
        if exceptions is None or not hasattr(exceptions, "YFRateLimitError"):
            return RuntimeError
        return exceptions.YFRateLimitError

    def _build_price_records(self, ticker: str, history: "pd.DataFrame") -> list[dict]:
        records = []
        for timestamp, row in history.iterrows():
            ts = timestamp.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            records.append(
                {
                    "id": f"{ticker}-{ts.isoformat()}",
                    "source": "yahoo_price",
                    "ticker": ticker,
                    "timestamp": ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "text": None,
                    "open": self._to_float(row.get("Open")),
                    "high": self._to_float(row.get("High")),
                    "low": self._to_float(row.get("Low")),
                    "close": self._to_float(row.get("Close")),
                    "volume": self._to_int(row.get("Volume")),
                    "score": None,
                    "url": None,
                }
            )
        return records

    def _build_news_records(
        self, ticker: str, news_items: list[dict[str, Any]], lookback_hours: int
    ) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        records = []
        if news_items is None:
            return records

        for item in news_items:
            published_ts = item.get("providerPublishTime")
            published = (
                datetime.fromtimestamp(published_ts, tz=timezone.utc)
                if published_ts
                else datetime.now(timezone.utc)
            )
            if published < cutoff:
                continue
            text = " ".join(part for part in [item.get("title"), item.get("summary")] if part)
            records.append(
                {
                    "id": item.get("uuid") or item.get("link") or f"{ticker}-{published_ts}",
                    "source": "yahoo_news",
                    "ticker": ticker,
                    "timestamp": published.isoformat().replace("+00:00", "Z"),
                    "text": text or None,
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": None,
                    "score": None,
                    "url": item.get("link"),
                }
            )
        return records

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None or pd.isna(value):
            return None
        return int(value)
