"""
Integration tests for data ingestion pipeline.

These tests call real external APIs and are skipped in CI by default.
Tag: @pytest.mark.integration
"""

import json

import pytest

from src.ingestion.alphavantage_scraper import AlphaVantageNewsScraper
from src.ingestion.news_rss import NewsRSSScraper
from src.ingestion.yahoo_finance import YahooFinanceScraper


@pytest.mark.integration
class TestYahooFinanceIntegration:
    """Integration tests for Yahoo Finance scraper using the real API."""

    def test_fetch_returns_price_records(self):
        scraper = YahooFinanceScraper()
        records = scraper.fetch("AAPL", lookback_hours=48)

        assert isinstance(records, list)
        assert len(records) > 0

        price_records = [r for r in records if r.get("source") == "yahoo_price"]
        assert len(price_records) > 0

        rec = price_records[0]
        assert rec["ticker"] == "AAPL"
        assert rec["close"] is not None
        assert rec["volume"] is not None

    def test_fetch_record_schema(self):
        scraper = YahooFinanceScraper()
        records = scraper.fetch("MSFT", lookback_hours=48)

        required_keys = {"id", "source", "ticker", "timestamp", "text"}
        for rec in records[:5]:
            assert required_keys.issubset(rec.keys()), f"Missing keys: {required_keys - rec.keys()}"

    def test_save_creates_json_file(self, tmp_path, monkeypatch):
        scraper = YahooFinanceScraper()
        records = scraper.fetch("AAPL", lookback_hours=48)

        if not records:
            pytest.skip("No records fetched")

        monkeypatch.chdir(tmp_path)
        scraper.save(records, source="yahoo_price", ticker="AAPL")
        files = list((tmp_path / "data" / "raw" / "yahoo_price").glob("AAPL_*.json"))
        assert len(files) > 0


@pytest.mark.integration
class TestNewsRSSIntegration:
    """Integration tests for News RSS scraper."""

    def test_fetch_returns_list(self):
        scraper = NewsRSSScraper()
        records = scraper.fetch("AAPL", lookback_hours=48)

        assert isinstance(records, list)

    def test_fetch_record_schema(self):
        scraper = NewsRSSScraper()
        records = scraper.fetch("AAPL", lookback_hours=48)

        required_keys = {"id", "source", "ticker", "timestamp"}
        for rec in records[:5]:
            assert required_keys.issubset(rec.keys())

    def test_deduplication(self, tmp_path, monkeypatch):
        """Running fetch twice should not return duplicate IDs."""
        monkeypatch.chdir(tmp_path)
        scraper = NewsRSSScraper()
        first = scraper.fetch("AAPL", lookback_hours=48)
        second = scraper.fetch("AAPL", lookback_hours=48)

        first_ids = {r["id"] for r in first}
        second_ids = {r["id"] for r in second}
        assert len(second_ids - first_ids) == 0, "Second run returned new IDs — dedup failed"


@pytest.mark.integration
class TestAlphaVantageIntegration:
    """Integration tests for Alpha Vantage fundamentals scraper."""

    def test_fetch_returns_list_even_without_key(self):
        """Without ALPHAVANTAGE_API_KEY, fetch should return [] not raise."""
        import os

        env_key = os.environ.pop("ALPHAVANTAGE_API_KEY", None)
        try:
            scraper = AlphaVantageNewsScraper()
            result = scraper.fetch("AAPL", lookback_hours=24)
            assert isinstance(result, list)
            assert len(result) == 0
        finally:
            if env_key:
                os.environ["ALPHAVANTAGE_API_KEY"] = env_key

    def test_fetch_with_key_returns_fundamentals(self):
        import os

        if not os.getenv("ALPHAVANTAGE_API_KEY"):
            pytest.skip("ALPHAVANTAGE_API_KEY not set")

        scraper = AlphaVantageNewsScraper()
        records = scraper.fetch("AAPL", lookback_hours=24)

        assert isinstance(records, list)
        if records:
            rec = records[0]
            assert rec["ticker"] == "AAPL"
            assert rec["source"] == "alphavantage_fundamentals"


@pytest.mark.integration
class TestIngestionPipeline:
    """Integration tests for multi-ticker ingestion."""

    def test_multiple_tickers(self):
        tickers = ["AAPL", "MSFT"]
        scraper = YahooFinanceScraper()

        all_records = {}
        for ticker in tickers:
            records = scraper.fetch(ticker, lookback_hours=48)
            all_records[ticker] = records
            assert isinstance(records, list), f"Expected list for {ticker}"

        # At least one ticker should have data
        total = sum(len(v) for v in all_records.values())
        assert total > 0, "No records fetched for any ticker"

    def test_news_json_serializable(self):
        scraper = NewsRSSScraper()
        records = scraper.fetch("AAPL", lookback_hours=48)

        for rec in records[:5]:
            serialized = json.dumps(rec)
            loaded = json.loads(serialized)
            assert loaded["ticker"] == rec["ticker"]
