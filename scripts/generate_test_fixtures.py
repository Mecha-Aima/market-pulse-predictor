#!/usr/bin/env python3
"""
Generate test fixtures for integration testing.

This script creates small, realistic test datasets that can be used
for integration tests without requiring real API calls or large data files.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Fixture paths
FIXTURES_DIR = Path("tests/fixtures")
RAW_DIR = FIXTURES_DIR / "raw"
PROCESSED_DIR = FIXTURES_DIR / "processed"
FEATURES_DIR = FIXTURES_DIR / "features"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_price_data(ticker: str, days: int = 30) -> pd.DataFrame:
    """Generate synthetic price data for a ticker."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price movements
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, days)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add OHLCV data
    data = {
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': prices * (1 + np.random.uniform(0.005, 0.02, days)),
        'Low': prices * (1 - np.random.uniform(0.005, 0.02, days)),
        'Close': prices,
        'Volume': np.random.randint(1_000_000, 10_000_000, days),
        'Ticker': ticker
    }
    
    df = pd.DataFrame(data)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df


def generate_news_data(ticker: str, count: int = 20) -> list:
    """Generate synthetic news data."""
    news_templates = [
        "{ticker} reports strong quarterly earnings",
        "{ticker} announces new product launch",
        "Analysts upgrade {ticker} stock rating",
        "{ticker} faces regulatory challenges",
        "Market volatility affects {ticker} trading",
        "{ticker} CEO discusses future strategy",
        "Investors bullish on {ticker} prospects",
        "{ticker} stock reaches new high",
        "Concerns raised about {ticker} valuation",
        "{ticker} expands into new markets"
    ]
    
    news = []
    base_date = datetime.now()
    
    for i in range(count):
        date = base_date - timedelta(days=i)
        template = np.random.choice(news_templates)
        
        news.append({
            'title': template.format(ticker=ticker),
            'published': date.isoformat(),
            'source': np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'WSJ']),
            'url': f'https://example.com/news/{i}',
            'ticker': ticker,
            'text': f"Article about {ticker}. " * 10  # Short article text
        })
    
    return news



def generate_stocktwits_data(ticker: str, count: int = 15) -> list:
    """Generate synthetic StockTwits data."""
    sentiments = ['Bullish', 'Bearish']
    
    messages = []
    base_date = datetime.now()
    
    for i in range(count):
        date = base_date - timedelta(hours=i)
        
        messages.append({
            'id': i,
            'body': f'Message about ${ticker}',
            'created_at': date.isoformat(),
            'user': {'username': f'user_{i}'},
            'symbols': [{'symbol': ticker}],
            'entities': {
                'sentiment': {
                    'basic': np.random.choice(sentiments)
                }
            }
        })
    
    return messages


def generate_sentiment_data(ticker: str, count: int = 30) -> pd.DataFrame:
    """Generate synthetic sentiment analysis results."""
    dates = pd.date_range(end=datetime.now(), periods=count, freq='D')
    
    data = {
        'date': dates,
        'ticker': ticker,
        'text': [f'Sample text about {ticker}'] * count,
        'source': np.random.choice(['news_rss', 'finnhub', 'stocktwits'], count),
        'vader_compound': np.random.uniform(-1, 1, count),
        'vader_pos': np.random.uniform(0, 1, count),
        'vader_neu': np.random.uniform(0, 1, count),
        'vader_neg': np.random.uniform(0, 1, count),
    }
    
    df = pd.DataFrame(data)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df


def generate_features_data(ticker: str, count: int = 25) -> pd.DataFrame:
    """Generate synthetic feature data."""
    dates = pd.date_range(end=datetime.now(), periods=count, freq='D')
    
    # Price features
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, count)))
    
    data = {
        'date': dates,
        'ticker': ticker,
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, count),
        'returns': np.random.normal(0.001, 0.02, count),
        'volatility': np.random.uniform(0.01, 0.05, count),
        'ma_7': prices * (1 + np.random.uniform(-0.02, 0.02, count)),
        'ma_30': prices * (1 + np.random.uniform(-0.05, 0.05, count)),
        'sentiment_mean': np.random.uniform(-0.5, 0.5, count),
        'sentiment_count': np.random.randint(0, 50, count),
        'label': np.random.choice([0, 1], count)  # Binary classification
    }
    
    df = pd.DataFrame(data)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df


def main():
    """Generate all test fixtures."""
    print("Generating test fixtures...")
    print("=" * 50)
    
    # Test tickers
    tickers = ['AAPL', 'MSFT']
    
    total_size = 0
    
    for ticker in tickers:
        print(f"\nGenerating fixtures for {ticker}...")
        
        # Raw data
        print("  • Price data...")
        price_df = generate_price_data(ticker, days=30)
        price_path = RAW_DIR / f"yahoo_{ticker}.parquet"
        price_df.to_parquet(price_path, index=False)
        total_size += price_path.stat().st_size
        
        print("  • News data...")
        news_data = generate_news_data(ticker, count=20)
        news_path = RAW_DIR / f"news_{ticker}.json"
        with open(news_path, 'w') as f:
            json.dump(news_data, f, indent=2)
        total_size += news_path.stat().st_size
        
        print("  • StockTwits data...")
        stocktwits_data = generate_stocktwits_data(ticker, count=15)
        stocktwits_path = RAW_DIR / f"stocktwits_{ticker}.json"
        with open(stocktwits_path, 'w') as f:
            json.dump(stocktwits_data, f, indent=2)
        total_size += stocktwits_path.stat().st_size
        
        # Processed data
        print("  • Sentiment data...")
        sentiment_df = generate_sentiment_data(ticker, count=30)
        sentiment_path = PROCESSED_DIR / f"sentiment_{ticker}.parquet"
        sentiment_df.to_parquet(sentiment_path, index=False)
        total_size += sentiment_path.stat().st_size
        
        # Features
        print("  • Feature data...")
        features_df = generate_features_data(ticker, count=25)
        features_path = FEATURES_DIR / f"features_{ticker}.parquet"
        features_df.to_parquet(features_path, index=False)
        total_size += features_path.stat().st_size
    
    # Create metadata file
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'tickers': tickers,
        'files': {
            'raw': [
                f'yahoo_{t}.parquet' for t in tickers
            ] + [
                f'stocktwits_{t}.json' for t in tickers
            ],
            'processed': [
                f'sentiment_{t}.parquet' for t in tickers
            ],
            'features': [
                f'features_{t}.parquet' for t in tickers
            ]
        },
        'total_size_bytes': total_size,
        'total_size_mb': round(total_size / (1024 * 1024), 2)
    }
    
    metadata_path = FIXTURES_DIR / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 50)
    print("✓ Test fixtures generated successfully!")
    print(f"  Total size: {metadata['total_size_mb']} MB")
    print(f"  Location: {FIXTURES_DIR}")
    print(f"  Tickers: {', '.join(tickers)}")
    total_files = (
        len(metadata['files']['raw'])
        + len(metadata['files']['processed'])
        + len(metadata['files']['features'])
    )
    print(f"  Files: {total_files}")


if __name__ == '__main__':
    main()
