from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

try:
    import praw
except ImportError:  # pragma: no cover
    praw = None

from src.ingestion.base_scraper import BaseScraper


class RedditScraper(BaseScraper):
    source_name = "reddit"
    per_ticker = False
    SUBREDDITS = ("stocks", "investing", "wallstreetbets", "finance", "StockMarket")
    POST_LIMIT = 25
    COMMENT_LIMIT = 5

    def fetch(self, ticker: str, lookback_hours: int) -> list[dict]:
        client = self._build_client()
        seen_ids = self._load_seen_ids(self.source_name)
        records: list[dict] = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        for subreddit_name in self.SUBREDDITS:
            subreddit = client.subreddit(subreddit_name)
            for post in subreddit.hot(limit=self.POST_LIMIT):
                created_at = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                if created_at < cutoff:
                    continue
                records.extend(self._submission_records(post, seen_ids))
        self._save_seen_ids(self.source_name, seen_ids)
        return records

    def _build_client(self):
        if praw is None:
            return SimpleNamespace(subreddit=lambda _: SimpleNamespace(hot=lambda limit: []))
        
        # Check if credentials are configured
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        username = os.getenv("REDDIT_USERNAME")
        password = os.getenv("REDDIT_PASSWORD")
        
        if not all([client_id, client_secret, username, password]):
            print("⚠️  Reddit credentials not configured - skipping Reddit scraper")
            return SimpleNamespace(subreddit=lambda _: SimpleNamespace(hot=lambda limit: []))
        
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent=os.getenv("REDDIT_USER_AGENT", "market-pulse-predictor/1.0"),
        )

    def _submission_records(self, post: Any, seen_ids: set[str]) -> list[dict]:
        records: list[dict] = []
        if post.id not in seen_ids:
            seen_ids.add(post.id)
            text = " ".join(
                part for part in [post.title, getattr(post, "selftext", "")] if part
            ).strip()
            records.extend(
                self._expand_tickers(
                    source=self.source_name,
                    record_id=post.id,
                    text=text,
                    timestamp=datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                    score=getattr(post, "upvote_ratio", None),
                    url=self._absolute_url(getattr(post, "permalink", None)),
                )
            )
        post.comments.replace_more(limit=0)
        for comment in post.comments.list()[: self.COMMENT_LIMIT]:
            if comment.id in seen_ids:
                continue
            seen_ids.add(comment.id)
            records.extend(
                self._expand_tickers(
                    source="reddit_comment",
                    record_id=comment.id,
                    text=getattr(comment, "body", None),
                    timestamp=datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
                    score=getattr(comment, "score", None),
                    url=self._absolute_url(getattr(comment, "permalink", None)),
                )
            )
        return records

    def _expand_tickers(
        self,
        *,
        source: str,
        record_id: str,
        text: str | None,
        timestamp: datetime,
        score: float | int | None,
        url: str | None,
    ) -> list[dict]:
        matched_tickers = self._match_tickers(text or "")
        return [
            {
                "id": record_id,
                "source": source,
                "ticker": ticker,
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "text": text,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
                "score": score,
                "url": url,
            }
            for ticker in matched_tickers
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

    @staticmethod
    def _absolute_url(permalink: str | None) -> str | None:
        if not permalink:
            return None
        return f"https://reddit.com{permalink}"
