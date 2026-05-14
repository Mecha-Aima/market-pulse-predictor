import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from airflow import DAG

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "feature_dag",
    default_args=default_args,
    description="Build time-series features",
    schedule_interval="15 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["features"],
)


def build_features():
    from src.features.run_features import main

    main()


build_features_task = PythonOperator(
    task_id="build_features",
    python_callable=build_features,
    dag=dag,
)

dvc_add_features_task = BashOperator(
    task_id="dvc_add_features",
    bash_command=(
        "cd /app && dvc add data/features/ "
        "&& git add data/features.dvc "
        '&& git commit -m "Update features" || true'
    ),
    dag=dag,
)

build_features_task >> dvc_add_features_task
