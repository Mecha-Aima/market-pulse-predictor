from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sentiment_dag',
    default_args=default_args,
    description='Run sentiment analysis on raw data',
    schedule_interval='5 * * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['sentiment'],
)


def run_sentiment_analysis():
    """Run sentiment analysis"""
    from src.sentiment.run_sentiment import main
    main()


run_sentiment_task = PythonOperator(
    task_id='run_sentiment',
    python_callable=run_sentiment_analysis,
    dag=dag,
)

dvc_add_processed_task = BashOperator(
    task_id='dvc_add_processed',
    bash_command='cd /app && dvc add data/processed/ && git add data/processed.dvc && git commit -m "Update processed data" || true',
    dag=dag,
)

# Set dependencies
run_sentiment_task >> dvc_add_processed_task
