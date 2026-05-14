from pathlib import Path
from typing import Any, Dict

import pytest

from tests.fixtures import (
    get_available_tickers,
    load_all_fixtures,
    load_features_data,
    load_news_data,
    load_price_data,
    load_sentiment_data,
    load_stocktwits_data,
)

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)

REQUIRED_ENV_KEYS = {
    "FINNHUB_API_KEY",
    "ALPHAVANTAGE_API_KEY",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "DVC_REMOTE_BUCKET",
    "AIRFLOW__CORE__EXECUTOR",
    "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
    "AIRFLOW__CORE__FERNET_KEY",
    "AIRFLOW__WEBSERVER__SECRET_KEY",
    "API_HOST",
    "API_PORT",
    "MODEL_REGISTRY_PATH",
    "TARGET_TICKERS",
}


@pytest.fixture()
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def test_ticker() -> str:
    """Provide a default test ticker."""
    return "AAPL"


@pytest.fixture()
def test_tickers() -> list:
    """Provide list of available test tickers."""
    return get_available_tickers()


@pytest.fixture()
def fixture_price_data(test_ticker: str):
    """Load price data fixture."""
    return load_price_data(test_ticker)


@pytest.fixture()
def fixture_news_data(test_ticker: str):
    """Load news data fixture."""
    return load_news_data(test_ticker)


@pytest.fixture()
def fixture_stocktwits_data(test_ticker: str):
    """Load StockTwits data fixture."""
    return load_stocktwits_data(test_ticker)


@pytest.fixture()
def fixture_sentiment_data(test_ticker: str):
    """Load sentiment data fixture."""
    return load_sentiment_data(test_ticker)


@pytest.fixture()
def fixture_features_data(test_ticker: str):
    """Load features data fixture."""
    return load_features_data(test_ticker)


@pytest.fixture()
def all_fixtures(test_ticker: str) -> Dict[str, Any]:
    """Load all fixtures for a ticker."""
    return load_all_fixtures(test_ticker)
