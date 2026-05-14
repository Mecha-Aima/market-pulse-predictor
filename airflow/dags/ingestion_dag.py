import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from airflow import DAG

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.finnhub_scraper import FinnhubNewsScraper
from src.ingestion.news_rss import NewsRSSScraper
from src.ingestion.stocktwits_scraper import StockTwitsScraper
from src.ingestion.yahoo_finance import YahooFinanceScraper

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Run once daily after market close (21:00 UTC = 5pm ET)
dag = DAG(
    "ingestion_dag",
    default_args=default_args,
    description="Daily ingestion of price and sentiment data",
    schedule_interval="0 21 * * 1-5",  # Mon-Fri at 21:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion"],
)

TICKERS = os.getenv("TARGET_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA,SPY").split(",")
# Lookback 48h to capture any data missed by the previous run
LOOKBACK_HOURS = 48


def ingest_yahoo() -> None:
    """Fetch daily OHLCV bars + Yahoo news for each ticker."""
    scraper = YahooFinanceScraper()
    for ticker in TICKERS:
        records = scraper.fetch(ticker, lookback_hours=LOOKBACK_HOURS)
        if records:
            scraper.save(records, source="yahoo_price", ticker=ticker)


def ingest_news_rss() -> None:
    """Fetch RSS headlines for each ticker."""
    scraper = NewsRSSScraper()
    for ticker in TICKERS:
        records = scraper.fetch(ticker, lookback_hours=LOOKBACK_HOURS)
        if records:
            scraper.save(records, source="news_rss", ticker=ticker)


def ingest_stocktwits() -> None:
    """Fetch StockTwits messages for each ticker."""
    scraper = StockTwitsScraper()
    for ticker in TICKERS:
        records = scraper.fetch(ticker, lookback_hours=LOOKBACK_HOURS)
        if records:
            scraper.save(records, source="stocktwits", ticker=ticker)


def ingest_finnhub() -> None:
    """Fetch Finnhub news/sentiment for each ticker."""
    scraper = FinnhubNewsScraper()
    for ticker in TICKERS:
        records = scraper.fetch(ticker, lookback_hours=LOOKBACK_HOURS)
        if records:
            scraper.save(records, source="finnhub", ticker=ticker)


# --- Tasks ---
ingest_yahoo_task = PythonOperator(
    task_id="ingest_yahoo",
    python_callable=ingest_yahoo,
    dag=dag,
)

ingest_news_rss_task = PythonOperator(
    task_id="ingest_news_rss",
    python_callable=ingest_news_rss,
    dag=dag,
)

ingest_stocktwits_task = PythonOperator(
    task_id="ingest_stocktwits",
    python_callable=ingest_stocktwits,
    dag=dag,
)

ingest_finnhub_task = PythonOperator(
    task_id="ingest_finnhub",
    python_callable=ingest_finnhub,
    dag=dag,
)

dvc_add_raw_task = BashOperator(
    task_id="dvc_add_raw",
    bash_command=(
        "cd /app && dvc add data/raw/ && git add data/raw.dvc "
        '&& git commit -m "chore: update raw data [skip ci]" || true'
    ),
    dag=dag,
)

# All ingestion tasks run in parallel, then DVC snapshots raw data
[
    ingest_yahoo_task,
    ingest_news_rss_task,
    ingest_stocktwits_task,
    ingest_finnhub_task,
] >> dvc_add_raw_task
