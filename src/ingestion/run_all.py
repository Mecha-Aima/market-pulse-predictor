from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

# Load .env before any scraper reads os.getenv()
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:  # pragma: no cover
    pass  # python-dotenv not installed; rely on shell environment

from src.ingestion.alphavantage_scraper import AlphaVantageNewsScraper
from src.ingestion.finnhub_scraper import FinnhubNewsScraper
from src.ingestion.news_rss import NewsRSSScraper
from src.ingestion.stocktwits_scraper import StockTwitsScraper
from src.ingestion.technicals import TechnicalsEnricher
from src.ingestion.yahoo_finance import YahooFinanceScraper

SCRAPER_CLASSES = [
    YahooFinanceScraper,
    FinnhubNewsScraper,
    NewsRSSScraper,
    StockTwitsScraper,
    AlphaVantageNewsScraper,  # RSI + MACD technical indicators (25 calls/day, runs last)
]


def get_target_tickers() -> list[str]:
    return [
        item.strip().upper() for item in os.getenv("TARGET_TICKERS", "").split(",") if item.strip()
    ]


def run_all_sources(lookback_hours: int) -> dict[str, int]:
    tickers = get_target_tickers()

    if not tickers:
        print(
            "⚠️  TARGET_TICKERS is empty — per-ticker scrapers will be skipped entirely.\n"
            "    Set TARGET_TICKERS in your .env file, e.g.:\n"
            "      TARGET_TICKERS=AAPL,MSFT,GOOGL,AMZN,TSLA,SPY"
        )

    print(f"ℹ️  Tickers: {tickers or '(none — per-ticker scrapers skipped)'}")
    print(f"ℹ️  Lookback: {lookback_hours}h\n")

    results: dict[str, int] = defaultdict(int)
    record_counts: dict[str, int] = defaultdict(int)
    errors: dict[str, str] = {}

    for scraper_class in SCRAPER_CLASSES:
        scraper_name = scraper_class.__name__
        try:
            scraper = scraper_class()
            targets = tickers if scraper.per_ticker else ["MARKET"]
            if not targets:
                print(f"  ⏭  {scraper_name}: skipped (no tickers configured)")
                continue
            for ticker in targets:
                try:
                    records = scraper.fetch(ticker, lookback_hours)
                    if records:
                        scraper.save(records, scraper.source_name, ticker)
                        results[scraper.source_name] += 1
                        record_counts[scraper.source_name] += len(records)
                        print(f"  ✓ {scraper_name} [{ticker}]: {len(records)} records")
                    else:
                        print(f"  ○ {scraper_name} [{ticker}]: 0 records returned")
                except Exception as e:
                    error_msg = f"{scraper_name} failed for {ticker}: {type(e).__name__}: {e}"
                    errors[f"{scraper_name}_{ticker}"] = error_msg
                    print(f"  ✗ {error_msg}")
        except Exception as e:
            error_msg = f"{scraper_name} initialization failed: {type(e).__name__}: {e}"
            errors[scraper_name] = error_msg
            print(f"  ✗ {error_msg}")

    # ── Local technicals (computed from saved Yahoo OHLCV — no API call) ──
    print("  ⚙  TechnicalsEnricher: computing RSI, MACD, BB, ATR, OBV from OHLCV...")
    try:
        tech_counts = TechnicalsEnricher().run()
        for ticker, count in tech_counts.items():
            record_counts["technicals"] += count
            results["technicals"] += 1
    except Exception as e:
        errors["TechnicalsEnricher"] = str(e)
        print(f"  ✗ TechnicalsEnricher failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"✓ {len(results)} source(s) produced data:")
    for source, count in record_counts.items():
        print(f"  • {source}: {count} records across {results[source]} ticker(s)")
    if errors:
        print(f"⚠️  {len(errors)} error(s) (pipeline continued despite errors)")
    print("="*60)

    return dict(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-hours", type=int, default=24)
    args = parser.parse_args()
    run_all_sources(lookback_hours=args.lookback_hours)


if __name__ == "__main__":
    main()
