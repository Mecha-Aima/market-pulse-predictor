from __future__ import annotations

from time import sleep

import httpx

from src.ingestion.base_scraper import BaseScraper


class StockTwitsScraper(BaseScraper):
    source_name = "stocktwits"

    def fetch(self, ticker: str, lookback_hours: int) -> list[dict]:
        seen_ids = self._load_seen_ids(self.source_name)
        response = httpx.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
        records = []
        for message in payload.get("messages", []):
            message_id = str(message["id"])
            if message_id in seen_ids:
                continue
            seen_ids.add(message_id)
            sentiment = message.get("entities", {}).get("sentiment") or {}
            records.append(
                {
                    "id": message_id,
                    "source": self.source_name,
                    "ticker": ticker,
                    "timestamp": message["created_at"],
                    "text": message.get("body"),
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": None,
                    "score": sentiment.get("basic"),
                    "url": message.get("url"),
                }
            )
        self._save_seen_ids(self.source_name, seen_ids)
        sleep(1)
        return records

