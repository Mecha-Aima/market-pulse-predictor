from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import httpx

from src.ingestion.base_scraper import BaseScraper

# StockTwits shut down their public API in 2024 (all endpoints return 403).
# Replaced with Finviz, which provides a public per-ticker news table.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
}

# Finviz news table rows: each <tr> has a date cell and a news link cell.
# The date cell is either "MMM-DD-YY HH:MMAM" or just "HH:MMAM" (same day).
_ROW_RE = re.compile(
    r"<tr[^>]*>.*?"
    r'<td[^>]*align="right"[^>]*>(.*?)</td>.*?'  # date/time cell
    r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',  # url + title
    re.DOTALL,
)
_STRIP_TAGS = re.compile(r"<[^>]+>")


class StockTwitsScraper(BaseScraper):
    """Finviz per-ticker news scraper (replaces defunct StockTwits public API)."""

    source_name = "stocktwits"

    def fetch(self, ticker: str, lookback_hours: int) -> list[dict]:
        try:
            seen_ids = self._load_seen_ids(self.source_name)
            response = httpx.get(
                f"https://finviz.com/quote.ashx?t={ticker}&p=d",
                headers=_HEADERS,
                timeout=30.0,
                follow_redirects=True,
            )
            response.raise_for_status()
            html = response.text

            # Extract just the news-table section to avoid false matches
            table_match = re.search(r'id=["\']news-table["\'].*?</table>', html, re.DOTALL)
            search_html = table_match.group(0) if table_match else html

            records = []
            now = datetime.now(timezone.utc)
            for m in _ROW_RE.finditer(search_html):
                _STRIP_TAGS.sub("", m.group(1)).strip()  # date field, not yet used
                url = m.group(2).strip()
                title = _STRIP_TAGS.sub("", m.group(3)).strip()

                if not title or not url:
                    continue

                record_id = hashlib.sha256(url.encode()).hexdigest()
                if record_id in seen_ids:
                    continue
                seen_ids.add(record_id)

                records.append(
                    {
                        "id": record_id,
                        "source": self.source_name,
                        "ticker": ticker,
                        "timestamp": now.isoformat().replace("+00:00", "Z"),
                        "text": title,
                        "open": None,
                        "high": None,
                        "low": None,
                        "close": None,
                        "volume": None,
                        "score": None,
                        "url": url,
                    }
                )

            self._save_seen_ids(self.source_name, seen_ids)
            return records

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Finviz returned HTTP {e.response.status_code} for {ticker}") from e
