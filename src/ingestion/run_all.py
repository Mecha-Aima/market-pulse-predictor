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
    errors: dict[str, str] = {}
    
    for scraper_class in SCRAPER_CLASSES:
        scraper_name = scraper_class.__name__
        try:
            scraper = scraper_class()
            targets = tickers if scraper.per_ticker else ["MARKET"]
            for ticker in targets:
                try:
                    records = scraper.fetch(ticker, lookback_hours)
                    if records:
                        scraper.save(records, scraper.source_name, ticker)
                        results[scraper.source_name] += 1
                except Exception as e:
                    error_msg = f"{scraper_name} failed for {ticker}: {type(e).__name__}: {e}"
                    errors[f"{scraper_name}_{ticker}"] = error_msg
                    print(f"⚠️  {error_msg}")
        except Exception as e:
            error_msg = f"{scraper_name} initialization failed: {type(e).__name__}: {e}"
            errors[scraper_name] = error_msg
            print(f"⚠️  {error_msg}")
    
    # Print summary
    print(f"\n✓ Successfully fetched from {len(results)} sources")
    if errors:
        print(f"⚠️  {len(errors)} errors occurred (pipeline continued)")
    
    return dict(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-hours", type=int, default=24)
    args = parser.parse_args()
    run_all_sources(lookback_hours=args.lookback_hours)


if __name__ == "__main__":
    main()
