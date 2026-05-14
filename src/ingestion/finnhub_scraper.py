from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import httpx

from src.ingestion.base_scraper import BaseScraper

# Finnhub free tier: 60 req/min, no daily cap.
# Per-ticker company news with 1 year of history.
_BASE = "https://finnhub.io/api/v1"


class FinnhubNewsScraper(BaseScraper):
    """Fetches per-ticker company news from Finnhub (free tier: 60 req/min)."""

    source_name = "finnhub"
    per_ticker = True

    def fetch(self, ticker: str, lookback_hours: int) -> list[dict]:
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            print(
                "  ⏭  FinnhubNewsScraper: skipping — FINNHUB_API_KEY not set "
                "(get a free key at finnhub.io)"
            )
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        date_from = cutoff.strftime("%Y-%m-%d")
        date_to = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        seen_ids = self._load_seen_ids(self.source_name)

        response = httpx.get(
            f"{_BASE}/company-news",
            params={
                "symbol": ticker,
                "from": date_from,
                "to": date_to,
                "token": api_key,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        articles = response.json()

        if not isinstance(articles, list):
            # Finnhub returns {"error": "..."} on bad key
            raise RuntimeError(f"Finnhub unexpected response: {articles}")

        records = []
        for article in articles:
            article_id = str(article.get("id", "")) or article.get("url", "")
            if not article_id or article_id in seen_ids:
                continue
            seen_ids.add(article_id)

            published_ts = article.get("datetime")
            published = (
                datetime.fromtimestamp(published_ts, tz=timezone.utc)
                if published_ts
                else datetime.now(timezone.utc)
            )
            if published < cutoff:
                continue

            headline = article.get("headline", "")
            summary = article.get("summary", "")
            text = " ".join(p for p in [headline, summary] if p).strip() or None

            records.append(
                {
                    "id": article_id,
                    "source": self.source_name,
                    "ticker": ticker,
                    "timestamp": published.isoformat().replace("+00:00", "Z"),
                    "text": text,
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": None,
                    "score": article.get("sentiment"),  # Finnhub provides sentiment score
                    "url": article.get("url"),
                    "fh_category": article.get("category"),
                    "fh_source": article.get("source"),
                    "fh_related": article.get("related"),
                }
            )

        self._save_seen_ids(self.source_name, seen_ids)
        return records
