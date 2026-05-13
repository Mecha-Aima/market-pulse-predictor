from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.yahoo_finance import YahooFinanceScraper
from src.ingestion.reuters_rss import ReutersRSSScraper
from src.ingestion.reddit_scraper import RedditScraper
from src.ingestion.twitter_scraper import TwitterScraper

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ingestion_dag',
    default_args=default_args,
    description='Ingest data from all sources',
    schedule_interval='*/30 * * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ingestion'],
)


def ingest_yahoo():
    """Ingest data from Yahoo Finance"""
    scraper = YahooFinanceScraper()
    import os
    tickers = os.getenv('TARGET_TICKERS', 'AAPL,MSFT,GOOGL,AMZN,TSLA,SPY').split(',')
    for ticker in tickers:
        records = scraper.fetch(ticker, lookback_hours=24)
        if records:
            scraper.save(records, 'yahoo', ticker)


def ingest_reuters():
    """Ingest data from Reuters RSS"""
    scraper = ReutersRSSScraper()
    records = scraper.fetch('MARKET', lookback_hours=24)
    if records:
        scraper.save(records, 'reuters', 'MARKET')


def ingest_reddit():
    """Ingest data from Reddit"""
    scraper = RedditScraper()
    import os
    tickers = os.getenv('TARGET_TICKERS', 'AAPL,MSFT,GOOGL,AMZN,TSLA,SPY').split(',')
    for ticker in tickers:
        records = scraper.fetch(ticker, lookback_hours=24)
        if records:
            scraper.save(records, 'reddit', ticker)


def ingest_twitter():
    """Ingest data from Twitter"""
    scraper = TwitterScraper()
    import os
    tickers = os.getenv('TARGET_TICKERS', 'AAPL,MSFT,GOOGL,AMZN,TSLA,SPY').split(',')
    for ticker in tickers:
        records = scraper.fetch(ticker, lookback_hours=24)
        if records:
            scraper.save(records, 'twitter', ticker)


# Define tasks
ingest_yahoo_task = PythonOperator(
    task_id='ingest_yahoo',
    python_callable=ingest_yahoo,
    dag=dag,
)

ingest_reuters_task = PythonOperator(
    task_id='ingest_reuters',
    python_callable=ingest_reuters,
    dag=dag,
)

ingest_reddit_task = PythonOperator(
    task_id='ingest_reddit',
    python_callable=ingest_reddit,
    dag=dag,
)

ingest_twitter_task = PythonOperator(
    task_id='ingest_twitter',
    python_callable=ingest_twitter,
    dag=dag,
)

dvc_add_raw_task = BashOperator(
    task_id='dvc_add_raw',
    bash_command='cd /app && dvc add data/raw/ && git add data/raw.dvc && git commit -m "Update raw data" || true',
    dag=dag,
)

# Set dependencies: all ingestion tasks run in parallel, then DVC add
[ingest_yahoo_task, ingest_reuters_task, ingest_reddit_task, ingest_twitter_task] >> dvc_add_raw_task
