from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RAW_RECORD_KEYS = (
    "id",
    "source",
    "ticker",
    "timestamp",
    "text",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "score",
    "url",
    "av_sentiment_label",
    "av_sentiment_score",
)


class BaseScraper(ABC):
    source_name = "base"
    per_ticker = True

    @abstractmethod
    def fetch(self, ticker: str, lookback_hours: int) -> list[dict]:
        """Fetch raw records for a ticker."""

    def save(self, records: list[dict], source: str, ticker: str) -> Path:
        output_dir = self._raw_dir(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = output_dir / f"{ticker}_{timestamp}.json"
        normalized = [self._normalize_record(record) for record in records]
        output_path.write_text(json.dumps(normalized, indent=2))
        return output_path

    def load_latest(self, source: str, ticker: str) -> list[dict]:
        latest_file = max(self._raw_dir(source).glob(f"{ticker}_*.json"))
        return json.loads(latest_file.read_text())

    def _raw_dir(self, source: str) -> Path:
        return Path.cwd() / "data" / "raw" / source

    def _seen_ids_path(self, source: str) -> Path:
        return self._raw_dir(source) / "seen_ids.json"

    def _load_seen_ids(self, source: str) -> set[str]:
        path = self._seen_ids_path(source)
        if not path.exists():
            return set()
        return set(json.loads(path.read_text()))

    def _save_seen_ids(self, source: str, seen_ids: set[str]) -> None:
        path = self._seen_ids_path(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(seen_ids), indent=2))

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        required = {"id", "source", "ticker", "timestamp"}
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(f"Missing required raw record keys: {', '.join(missing)}")

        normalized = {
            key: record[key]
            for key in RAW_RECORD_KEYS
            if key in record
            or key in {"text", "open", "high", "low", "close", "volume", "score", "url"}  # noqa: E501
        }
        for optional_key in ("text", "open", "high", "low", "close", "volume", "score", "url"):
            normalized.setdefault(optional_key, None)
        extras = {key: value for key, value in record.items() if key not in RAW_RECORD_KEYS}
        return normalized | extras
