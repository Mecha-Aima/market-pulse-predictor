from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx

from src.ingestion.base_scraper import BaseScraper

LOGGER = logging.getLogger(__name__)

# AlphaVantage free tier: 25 req/day.
# Technical indicators (RSI, MACD) are PREMIUM endpoints — not available free.
# Free endpoints: TIME_SERIES_DAILY, OVERVIEW (fundamentals) — both redundant
# with Yahoo Finance which we already use.
#
# This class now fetches company OVERVIEW (fundamentals: P/E, EPS, market cap,
# 52-week range) — one call per ticker, 6 calls total, well within 25/day.
# RSI/MACD are computed locally in src/ingestion/technicals.py from Yahoo OHLCV.


class AlphaVantageNewsScraper(BaseScraper):
    """Fetches company fundamental overview from AlphaVantage (free endpoint)."""

    source_name = "alphavantage"
    per_ticker = True

    _FUNDAMENTAL_FIELDS = [
        "MarketCapitalization", "PERatio", "EPS", "DividendYield",
        "52WeekHigh", "52WeekLow", "50DayMovingAverage", "200DayMovingAverage",
        "Beta", "ProfitMargin", "RevenuePerShareTTM",
    ]

    def fetch(self, ticker: str, lookback_hours: int) -> list[dict]:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return []

        try:
            response = httpx.get(
                "https://www.alphavantage.co/query",
                params={"function": "OVERVIEW", "symbol": ticker, "apikey": api_key},
                timeout=30.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            LOGGER.warning("AlphaVantage OVERVIEW request failed for %s: %s", ticker, e)
            return []

        # Rate limit / access error detection
        if "Information" in payload or "Note" in payload:
            msg = payload.get("Information") or payload.get("Note", "rate limit")
            LOGGER.warning("AlphaVantage rate-limited for %s: %s", ticker, msg[:80])
            return []  # Soft-fail — don't crash the whole pipeline

        if not payload.get("Symbol"):
            LOGGER.debug("AlphaVantage OVERVIEW empty for %s. Keys: %s", ticker, list(payload.keys()))
            return []

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        record = {
            "id": f"{ticker}-fundamentals-{datetime.now(timezone.utc).date()}",
            "source": "alphavantage_fundamentals",
            "ticker": ticker,
            "timestamp": now,
            "text": None,
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": None,
            "score": None,
            "url": None,
        }
        for field in self._FUNDAMENTAL_FIELDS:
            val = payload.get(field)
            safe_key = f"av_{field.lower()}"
            record[safe_key] = self._to_float(val) if val not in (None, "None", "-", "") else None

        return [record]

    @staticmethod
    def _to_float(value) -> float | None:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
