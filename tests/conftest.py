from pathlib import Path

import pytest

from src.api.main import app

REQUIRED_ENV_KEYS = {
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USERNAME",
    "REDDIT_PASSWORD",
    "REDDIT_USER_AGENT",
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
def api_app():
    return app
