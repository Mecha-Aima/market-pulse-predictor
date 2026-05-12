"""
Integration tests for data ingestion pipeline.

Tests end-to-end ingestion using real APIs with test tickers.
"""

import pytest
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta

from src.ingestion.yahoo_finance import YahooFinanceScraper
from src.ingestion.news_rss import NewsRSSScraper
from src.ingestion.alphavantage_scraper import AlphaVantageNewsScraper


@pytest.mark.integration
class TestYahooFinanceIntegration:
    """Integration tests for Yahoo Finance scraper."""
    
    def test_fetch_real_price_data(self, tmp_path):
        """Test: End-to-end ingestion of real price data."""
        scraper = YahooFinanceScraper(output_dir=str(tmp_path))
        
        # Use a stable test ticker
        ticker = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Fetch real data
        data = scraper.fetch(
            ticker=ticker,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        # Verify data structure
        assert isinstance(data, list), "Should return list of records"
        assert len(data) > 0, "Should fetch at least some data"
        
        # Verify record structure
        record = data[0]
        required_fields = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
        for field in required_fields:
            assert field in record, f"Record should have {field} field"
        
        # Verify data types
        assert isinstance(record['Close'], (int, float)), "Close should be numeric"
        assert isinstance(record['Volume'], (int, float)), "Volume should be numeric"
        assert record['Ticker'] == ticker, "Ticker should match"
    
    def test_save_price_data(self, tmp_path):
        """Test: Save fetched price data to parquet."""
        scraper = YahooFinanceScraper(output_dir=str(tmp_path))
        
        ticker = "MSFT"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        # Fetch and save
        data = scraper.fetch(
            ticker=ticker,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        scraper.save(data, ticker)
        
        # Verify file exists
        output_file = tmp_path / f"yahoo_{ticker}.parquet"
        assert output_file.exists(), "Output file should be created"
        
        # Verify file content
        df = pd.read_parquet(output_file)
        assert len(df) > 0, "File should contain data"
        assert 'Close' in df.columns, "Should have Close column"


@pytest.mark.integration
class TestNewsRSSIntegration:
    """Integration tests for News RSS scraper."""
    
    def test_fetch_real_news(self, tmp_path):
        """Test: End-to-end ingestion of real news data."""
        scraper = NewsRSSScraper(output_dir=str(tmp_path))
        
        # Fetch news for a test ticker
        ticker = "AAPL"
        data = scraper.fetch(ticker=ticker, max_articles=5)
        
        # Verify data structure
        assert isinstance(data, list), "Should return list of articles"
        
        if len(data) > 0:  # News might not always be available
            article = data[0]
            required_fields = ['title', 'published', 'source', 'ticker']
            for field in required_fields:
                assert field in article, f"Article should have {field} field"
            
            assert article['ticker'] == ticker, "Ticker should match"
    
    def test_save_news_data(self, tmp_path):
        """Test: Save fetched news data to JSON."""
        scraper = NewsRSSScraper(output_dir=str(tmp_path))
        
        ticker = "GOOGL"
        data = scraper.fetch(ticker=ticker, max_articles=3)
        
        if len(data) > 0:
            scraper.save(data, ticker)
            
            # Verify file exists
            output_file = tmp_path / f"news_{ticker}.json"
            assert output_file.exists(), "Output file should be created"
            
            # Verify file content
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            assert len(saved_data) > 0, "File should contain data"
            assert saved_data[0]['ticker'] == ticker, "Ticker should match"


@pytest.mark.integration
class TestAlphaVantageIntegration:
    """Integration tests for Alpha Vantage scraper."""
    
    def test_fetch_real_news_with_sentiment(self, tmp_path):
        """Test: End-to-end ingestion of news with sentiment from Alpha Vantage."""
        scraper = AlphaVantageNewsScraper(output_dir=str(tmp_path))
        
        # Use a test ticker
        ticker = "AAPL"
        data = scraper.fetch(ticker=ticker, lookback_hours=24)
        
        # Verify data structure
        assert isinstance(data, list), "Should return list of articles"
        
        if len(data) > 0:  # API might have rate limits
            article = data[0]
            required_fields = ['ticker', 'text', 'timestamp']
            for field in required_fields:
                assert field in article, f"Article should have {field} field"
            
            # Alpha Vantage provides sentiment scores
            if 'av_sentiment_score' in article:
                assert isinstance(article['av_sentiment_score'], (int, float, type(None))), \
                    "Sentiment score should be numeric or None"
    
    def test_save_alphavantage_data(self, tmp_path):
        """Test: Save Alpha Vantage data to JSON."""
        scraper = AlphaVantageNewsScraper(output_dir=str(tmp_path))
        
        ticker = "MSFT"
        data = scraper.fetch(ticker=ticker, lookback_hours=24)
        
        if len(data) > 0:
            scraper.save(data, ticker)
            
            # Verify file exists
            output_file = tmp_path / f"alphavantage_{ticker}.json"
            assert output_file.exists(), "Output file should be created"
            
            # Verify file content
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            assert len(saved_data) > 0, "File should contain data"


@pytest.mark.integration
class TestIngestionPipeline:
    """Integration tests for complete ingestion pipeline."""
    
    def test_end_to_end_ingestion(self, tmp_path):
        """Test: Complete ingestion pipeline for a single ticker."""
        ticker = "AAPL"
        
        # 1. Fetch price data
        price_scraper = YahooFinanceScraper(output_dir=str(tmp_path))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        price_data = price_scraper.fetch(
            ticker=ticker,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        price_scraper.save(price_data, ticker)
        
        # 2. Fetch news data
        news_scraper = NewsRSSScraper(output_dir=str(tmp_path))
        news_data = news_scraper.fetch(ticker=ticker, max_articles=5)
        if len(news_data) > 0:
            news_scraper.save(news_data, ticker)
        
        # 3. Verify all data was collected
        price_file = tmp_path / f"yahoo_{ticker}.parquet"
        assert price_file.exists(), "Price data should be saved"
        
        # Verify price data quality
        df = pd.read_parquet(price_file)
        assert len(df) > 0, "Should have price records"
        assert df['Close'].notna().all(), "Close prices should not be null"
        assert (df['Volume'] >= 0).all(), "Volume should be non-negative"
        
        # If news was fetched, verify it
        news_file = tmp_path / f"news_{ticker}.json"
        if news_file.exists():
            with open(news_file, 'r') as f:
                news = json.load(f)
            assert len(news) > 0, "Should have news articles"
    
    def test_multiple_tickers_ingestion(self, tmp_path):
        """Test: Ingestion for multiple tickers."""
        tickers = ["AAPL", "MSFT"]
        
        price_scraper = YahooFinanceScraper(output_dir=str(tmp_path))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        for ticker in tickers:
            # Fetch and save price data
            data = price_scraper.fetch(
                ticker=ticker,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            price_scraper.save(data, ticker)
            
            # Verify file exists
            price_file = tmp_path / f"yahoo_{ticker}.parquet"
            assert price_file.exists(), f"Price data for {ticker} should be saved"
        
        # Verify all files were created
        parquet_files = list(tmp_path.glob("yahoo_*.parquet"))
        assert len(parquet_files) == len(tickers), \
            f"Should have {len(tickers)} price data files"
