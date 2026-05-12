from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import httpx

from src.ingestion.base_scraper import BaseScraper


class AlphaVantageNewsScraper(BaseScraper):
    source_name = "alphavantage"
    per_ticker = False

    def fetch(self, ticker: str, lookback_hours: int) -> list[dict]:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return []

        tickers = self._configured_tickers()
        if not tickers:
            return []

        response = httpx.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ",".join(tickers),
                "sort": "LATEST",
                "limit": 50,
                "apikey": api_key,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        records = []
        for article in payload.get("feed", []):
            published = self._parse_timestamp(article.get("time_published"))
            if published is not None and published < cutoff:
                continue
            for ticker_record in article.get("ticker_sentiment", []):
                matched_ticker = ticker_record.get("ticker")
                if matched_ticker not in tickers:
                    continue
                text = " ".join(
                    part for part in [article.get("title"), article.get("summary")] if part
                )
                records.append(
                    {
                        "id": article.get("url") or f"{matched_ticker}-{article.get('time_published')}",
                        "source": "alphavantage_news",
                        "ticker": matched_ticker,
                        "timestamp": (
                            published.isoformat().replace("+00:00", "Z")
                            if published is not None
                            else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                        ),
                        "text": text or None,
                        "open": None,
                        "high": None,
                        "low": None,
                        "close": None,
                        "volume": None,
                        "score": None,
                        "url": article.get("url"),
                        "av_sentiment_label": ticker_record.get("ticker_sentiment_label"),
                        "av_sentiment_score": self._to_float(
                            ticker_record.get("ticker_sentiment_score")
                        ),
                    }
                )
        return records

    def _configured_tickers(self) -> list[str]:
        return [
            item.strip().upper()
            for item in os.getenv("TARGET_TICKERS", "").split(",")
            if item.strip()
        ]

    @staticmethod
    def _parse_timestamp(raw_timestamp: str | None) -> datetime | None:
        if not raw_timestamp:
            return None
        return datetime.strptime(raw_timestamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

    @staticmethod
    def _to_float(value) -> float | None:
        if value is None:
            return None
        return float(value)

