from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    import feedparser
except ImportError:  # pragma: no cover
    feedparser = None

from src.ingestion.base_scraper import BaseScraper


class NewsRSSScraper(BaseScraper):
    source_name = "news_rss"
    per_ticker = False
    DEFAULT_FEEDS = [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
    ]

    def fetch(self, ticker: str, lookback_hours: int) -> list[dict]:
        if feedparser is None:
            raise RuntimeError("feedparser is required for RSS ingestion")

        seen_ids = self._load_seen_ids(self.source_name)
        records: list[dict] = []
        for url in self._feed_urls():
            feed = feedparser.parse(url)
            for entry in getattr(feed, "entries", []):
                entry_id = self._entry_id(entry)
                if entry_id in seen_ids or not self._is_recent(entry, lookback_hours):
                    continue
                seen_ids.add(entry_id)
                records.extend(self._entry_to_records(entry, entry_id))
        self._save_seen_ids(self.source_name, seen_ids)
        return records

    def _feed_urls(self) -> list[str]:
        override = os.getenv("NEWS_RSS_URLS")
        if override:
            return [item.strip() for item in override.split(",") if item.strip()]
        return self.DEFAULT_FEEDS

    def _entry_id(self, entry: dict[str, Any]) -> str:
        candidate = entry.get("id") or entry.get("link") or entry.get("title", "")
        if candidate == entry.get("link"):
            return hashlib.sha256(str(candidate).encode()).hexdigest()
        return str(candidate)

    def _is_recent(self, entry: dict[str, Any], lookback_hours: int) -> bool:
        published = entry.get("published_parsed")
        if not published:
            return True
        published_at = datetime(*published[:6], tzinfo=timezone.utc)
        return published_at >= datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    def _entry_to_records(self, entry: dict[str, Any], entry_id: str) -> list[dict]:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        text = " ".join(part for part in [title, summary] if part).strip() or None
        tickers = self._match_tickers(text or "")
        published = entry.get("published_parsed")
        if published:
            timestamp = datetime(*published[:6], tzinfo=timezone.utc)
        else:
            timestamp = datetime.fromtimestamp(time.time(), tz=timezone.utc)

        return [
            {
                "id": entry_id,
                "source": self.source_name,
                "ticker": matched_ticker,
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "text": text,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
                "score": None,
                "url": entry.get("link"),
            }
            for matched_ticker in tickers
        ]

    def _match_tickers(self, text: str) -> list[str]:
        configured = [
            item.strip().upper()
            for item in os.getenv("TARGET_TICKERS", "").split(",")
            if item.strip()
        ]
        matched = [
            ticker
            for ticker in configured
            if re.search(rf"\b{re.escape(ticker)}\b", text, flags=re.IGNORECASE)
        ]
        return matched or ["MARKET"]

