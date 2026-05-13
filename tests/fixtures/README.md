# Test Fixtures

This directory contains test fixtures for integration testing. Fixtures are small, synthetic datasets that simulate real data without requiring API calls or large files.

## Directory Structure

```
tests/fixtures/
├── raw/                    # Raw data fixtures (as scraped)
│   ├── yahoo_*.parquet    # Price data (OHLCV)
│   ├── news_*.json        # News articles
│   ├── reddit_*.json      # Reddit posts
│   └── stocktwits_*.json  # StockTwits messages
├── processed/              # Processed data fixtures
│   └── sentiment_*.parquet # Sentiment analysis results
├── features/               # Feature data fixtures
│   └── features_*.parquet  # Engineered features
├── models/                 # Model fixtures (if needed)
├── __init__.py            # Fixture loading utilities
├── metadata.json          # Fixture metadata
└── README.md              # This file
```

## Available Fixtures

### Tickers
- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation

### Data Types

#### 1. Price Data (`yahoo_*.parquet`)
OHLCV price data for 30 days.

**Columns:**
- `Date` (str): Date in YYYY-MM-DD format
- `Open` (float): Opening price
- `High` (float): High price
- `Low` (float): Low price
- `Close` (float): Closing price
- `Volume` (int): Trading volume
- `Ticker` (str): Stock ticker symbol

**Size:** ~5 KB per ticker

#### 2. News Data (`news_*.json`)
News articles for 20 days.

**Fields:**
- `title` (str): Article title
- `published` (str): Publication date (ISO format)
- `source` (str): News source (Reuters, Bloomberg, etc.)
- `url` (str): Article URL
- `ticker` (str): Related ticker
- `text` (str): Article text

**Size:** ~10 KB per ticker

#### 3. Reddit Data (`reddit_*.json`)
Reddit posts from r/wallstreetbets.

**Fields:**
- `id` (str): Post ID
- `title` (str): Post title
- `text` (str): Post body
- `score` (int): Upvotes
- `num_comments` (int): Comment count
- `created_utc` (int): Unix timestamp
- `subreddit` (str): Subreddit name
- `ticker` (str): Related ticker

**Size:** ~8 KB per ticker

#### 4. StockTwits Data (`stocktwits_*.json`)
StockTwits messages.

**Fields:**
- `id` (int): Message ID
- `body` (str): Message text
- `created_at` (str): Creation timestamp (ISO format)
- `user` (dict): User info
- `symbols` (list): Related symbols
- `entities` (dict): Sentiment info

**Size:** ~6 KB per ticker

#### 5. Sentiment Data (`sentiment_*.parquet`)
Sentiment analysis results for 30 days.

**Columns:**
- `date` (str): Date in YYYY-MM-DD format
- `ticker` (str): Stock ticker
- `text` (str): Analyzed text
- `source` (str): Data source (news, reddit, stocktwits)
- `vader_compound` (float): VADER compound score (-1 to 1)
- `vader_pos` (float): Positive score (0 to 1)
- `vader_neu` (float): Neutral score (0 to 1)
- `vader_neg` (float): Negative score (0 to 1)

**Size:** ~8 KB per ticker

#### 6. Features Data (`features_*.parquet`)
Engineered features for 25 days.

**Columns:**
- `date` (str): Date in YYYY-MM-DD format
- `ticker` (str): Stock ticker
- `close` (float): Closing price
- `volume` (int): Trading volume
- `returns` (float): Daily returns
- `volatility` (float): Price volatility
- `ma_7` (float): 7-day moving average
- `ma_30` (float): 30-day moving average
- `sentiment_mean` (float): Mean sentiment score
- `sentiment_count` (int): Number of sentiment records
- `label` (int): Binary label (0 or 1)

**Size:** ~6 KB per ticker

## Usage

### Loading Fixtures in Tests

```python
import pytest
from tests.fixtures import (
    load_price_data,
    load_news_data,
    load_sentiment_data,
    load_features_data,
    load_all_fixtures
)

def test_with_price_data():
    """Example test using price data fixture."""
    df = load_price_data('AAPL')
    assert len(df) == 30
    assert 'Close' in df.columns

def test_with_all_fixtures():
    """Example test using all fixtures."""
    fixtures = load_all_fixtures('AAPL')
    assert 'price' in fixtures
    assert 'sentiment' in fixtures
    assert 'features' in fixtures
```

### Using Pytest Fixtures

```python
def test_with_pytest_fixture(fixture_price_data):
    """Example using pytest fixture from conftest.py."""
    assert len(fixture_price_data) == 30

def test_with_all_fixtures(all_fixtures):
    """Example using all fixtures pytest fixture."""
    assert 'price' in all_fixtures
    assert 'features' in all_fixtures
```

### Available Pytest Fixtures

Defined in `tests/conftest.py`:

- `test_ticker` - Default test ticker ('AAPL')
- `test_tickers` - List of all available tickers
- `fixture_price_data` - Price data for default ticker
- `fixture_news_data` - News data for default ticker
- `fixture_reddit_data` - Reddit data for default ticker
- `fixture_stocktwits_data` - StockTwits data for default ticker
- `fixture_sentiment_data` - Sentiment data for default ticker
- `fixture_features_data` - Features data for default ticker
- `all_fixtures` - All fixtures for default ticker

## Regenerating Fixtures

To regenerate all fixtures:

```bash
python scripts/generate_test_fixtures.py
```

This will:
1. Create synthetic data for all tickers
2. Save fixtures to `tests/fixtures/`
3. Generate metadata.json
4. Report total size (should be < 10MB)

## Fixture Characteristics

### Data Quality
- **Realistic:** Synthetic data mimics real market data patterns
- **Reproducible:** Uses fixed random seed (42)
- **Small:** Total size < 1 MB for fast tests
- **Complete:** Covers all data pipeline stages

### Use Cases
- **Unit tests:** Test individual functions with small datasets
- **Integration tests:** Test data pipeline end-to-end
- **CI/CD:** Fast tests without external API calls
- **Development:** Local testing without credentials

## Best Practices

### When to Use Fixtures
✅ **Use fixtures for:**
- Unit tests
- Integration tests in CI/CD
- Testing data transformations
- Testing feature engineering
- Testing model training (small scale)

❌ **Don't use fixtures for:**
- Production data
- Performance benchmarking
- Real API integration tests (use test tickers instead)
- Large-scale model training

### Fixture Maintenance
- Regenerate fixtures when data schema changes
- Keep fixtures small (< 10MB total)
- Document any changes to fixture structure
- Version fixtures with code (committed to git)

## Metadata

Fixture metadata is stored in `metadata.json`:

```json
{
  "generated_at": "2026-05-12T...",
  "tickers": ["AAPL", "MSFT"],
  "files": {
    "raw": [...],
    "processed": [...],
    "features": [...]
  },
  "total_size_bytes": 81920,
  "total_size_mb": 0.08
}
```

## Adding New Fixtures

To add fixtures for a new ticker:

1. Edit `scripts/generate_test_fixtures.py`
2. Add ticker to `tickers` list
3. Run `python scripts/generate_test_fixtures.py`
4. Verify fixtures in `tests/fixtures/`

To add a new data type:

1. Add generation function to `scripts/generate_test_fixtures.py`
2. Add loading function to `tests/fixtures/__init__.py`
3. Add pytest fixture to `tests/conftest.py`
4. Document in this README

## Size Limits

- **Individual file:** < 100 KB
- **Total fixtures:** < 10 MB
- **Reason:** Fast CI/CD, quick git operations

If fixtures exceed limits:
1. Reduce number of records
2. Remove unnecessary columns
3. Use more efficient formats (parquet > json)

## Testing Fixtures

To verify fixtures are working:

```bash
# Test fixture loading
PYTHONPATH=. pytest tests/fixtures/ -v

# Test with fixtures
PYTHONPATH=. pytest tests/ --fixtures
```

## Troubleshooting

### FileNotFoundError
**Problem:** Fixture file not found

**Solution:**
```bash
python scripts/generate_test_fixtures.py
```

### Import Error
**Problem:** Cannot import fixture utilities

**Solution:**
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=.
pytest tests/
```

### Size Too Large
**Problem:** Fixtures exceed 10MB

**Solution:**
- Reduce number of records in `generate_test_fixtures.py`
- Remove unnecessary data types
- Use parquet instead of JSON

## Related Files

- `scripts/generate_test_fixtures.py` - Fixture generation script
- `tests/conftest.py` - Pytest fixture definitions
- `tests/fixtures/__init__.py` - Fixture loading utilities
- `.kiro/specs/phases-8-9-deployment/VALIDATION_REPORT.md` - Validation report

---

**Last Updated:** 2026-05-12  
**Total Size:** 0.08 MB  
**Tickers:** AAPL, MSFT  
**Files:** 12
