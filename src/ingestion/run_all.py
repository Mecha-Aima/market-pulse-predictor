from __future__ import annotations

import argparse
import os
from collections import defaultdict

from src.ingestion.alphavantage_scraper import AlphaVantageNewsScraper
from src.ingestion.news_rss import NewsRSSScraper
from src.ingestion.reddit_scraper import RedditScraper
from src.ingestion.stocktwits_scraper import StockTwitsScraper
from src.ingestion.yahoo_finance import YahooFinanceScraper

SCRAPER_CLASSES = [
    YahooFinanceScraper,
    NewsRSSScraper,
    RedditScraper,
    StockTwitsScraper,
    AlphaVantageNewsScraper,
]


def get_target_tickers() -> list[str]:
    return [
        item.strip().upper() for item in os.getenv("TARGET_TICKERS", "").split(",") if item.strip()
    ]


def run_all_sources(lookback_hours: int) -> dict[str, int]:
    tickers = get_target_tickers()
    results: dict[str, int] = defaultdict(int)
    for scraper_class in SCRAPER_CLASSES:
        scraper = scraper_class()
        targets = tickers if scraper.per_ticker else ["MARKET"]
        for ticker in targets:
            records = scraper.fetch(ticker, lookback_hours)
            if records:
                scraper.save(records, scraper.source_name, ticker)
                results[scraper.source_name] += 1
    return dict(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-hours", type=int, default=24)
    args = parser.parse_args()
    run_all_sources(lookback_hours=args.lookback_hours)


if __name__ == "__main__":
    main()
