import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml
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
    "training_dag",
    default_args=default_args,
    description="Train and evaluate models",
    schedule_interval="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training"],
)


def train_model(model_type: str):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    params["training"]["model"] = model_type
    with open("params.yaml", "w") as f:
        yaml.dump(params, f)
    from src.training.run_training import main

    main()


def train_rnn():
    train_model("rnn")


def train_lstm():
    train_model("lstm")


def train_gru():
    train_model("gru")


def evaluate_and_register():
    pass


train_rnn_task = PythonOperator(
    task_id="train_rnn",
    python_callable=train_rnn,
    dag=dag,
)

train_lstm_task = PythonOperator(
    task_id="train_lstm",
    python_callable=train_lstm,
    dag=dag,
)

train_gru_task = PythonOperator(
    task_id="train_gru",
    python_callable=train_gru,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id="evaluate_and_register",
    python_callable=evaluate_and_register,
    dag=dag,
)

[train_rnn_task, train_lstm_task, train_gru_task] >> evaluate_task
