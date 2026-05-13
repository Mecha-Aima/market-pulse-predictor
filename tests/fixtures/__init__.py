"""
Test fixtures module.

Provides utilities for loading test data fixtures for integration testing.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Fixture paths
FIXTURES_DIR = Path(__file__).parent
RAW_DIR = FIXTURES_DIR / "raw"
PROCESSED_DIR = FIXTURES_DIR / "processed"
FEATURES_DIR = FIXTURES_DIR / "features"


def load_metadata() -> Dict[str, Any]:
    """Load fixture metadata."""
    metadata_path = FIXTURES_DIR / "metadata.json"
    with open(metadata_path, 'r') as f:
        return json.load(f)


def load_price_data(ticker: str) -> pd.DataFrame:
    """
    Load price data fixture for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        DataFrame with OHLCV price data
    """
    path = RAW_DIR / f"yahoo_{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Price data fixture not found for {ticker}")
    return pd.read_parquet(path)


def load_news_data(ticker: str) -> List[Dict[str, Any]]:
    """
    Load news data fixture for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        List of news article dictionaries
    """
    path = RAW_DIR / f"news_{ticker}.json"
    if not path.exists():
        raise FileNotFoundError(f"News data fixture not found for {ticker}")
    with open(path, 'r') as f:
        return json.load(f)



def load_stocktwits_data(ticker: str) -> List[Dict[str, Any]]:
    """
    Load StockTwits data fixture for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        List of StockTwits message dictionaries
    """
    path = RAW_DIR / f"stocktwits_{ticker}.json"
    if not path.exists():
        raise FileNotFoundError(f"StockTwits data fixture not found for {ticker}")
    with open(path, 'r') as f:
        return json.load(f)


def load_sentiment_data(ticker: str) -> pd.DataFrame:
    """
    Load sentiment analysis data fixture for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        DataFrame with sentiment scores
    """
    path = PROCESSED_DIR / f"sentiment_{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Sentiment data fixture not found for {ticker}")
    return pd.read_parquet(path)


def load_features_data(ticker: str) -> pd.DataFrame:
    """
    Load feature data fixture for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        DataFrame with engineered features
    """
    path = FEATURES_DIR / f"features_{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature data fixture not found for {ticker}")
    return pd.read_parquet(path)


def get_available_tickers() -> List[str]:
    """
    Get list of tickers with available fixtures.
    
    Returns:
        List of ticker symbols
    """
    metadata = load_metadata()
    return metadata.get('tickers', [])


def fixture_exists(ticker: str, data_type: str) -> bool:
    """
    Check if a fixture exists for a ticker and data type.
    
    Args:
        ticker: Stock ticker symbol
        data_type: One of 'price', 'news', 'stocktwits', 'sentiment', 'features'
        
    Returns:
        True if fixture exists, False otherwise
    """
    paths = {
        'price': RAW_DIR / f"yahoo_{ticker}.parquet",
        'news': RAW_DIR / f"news_{ticker}.json",
        'stocktwits': RAW_DIR / f"stocktwits_{ticker}.json",
        'sentiment': PROCESSED_DIR / f"sentiment_{ticker}.parquet",
        'features': FEATURES_DIR / f"features_{ticker}.parquet",
    }
    
    path = paths.get(data_type)
    if path is None:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return path.exists()


# Convenience function for pytest fixtures
def load_all_fixtures(ticker: str) -> Dict[str, Any]:
    """
    Load all available fixtures for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with all fixture data
    """
    fixtures = {}
    
    if fixture_exists(ticker, 'price'):
        fixtures['price'] = load_price_data(ticker)
    
    if fixture_exists(ticker, 'news'):
        fixtures['news'] = load_news_data(ticker)
    
    if fixture_exists(ticker, 'stocktwits'):
        fixtures['stocktwits'] = load_stocktwits_data(ticker)
    
    if fixture_exists(ticker, 'sentiment'):
        fixtures['sentiment'] = load_sentiment_data(ticker)
    
    if fixture_exists(ticker, 'features'):
        fixtures['features'] = load_features_data(ticker)
    
    return fixtures
