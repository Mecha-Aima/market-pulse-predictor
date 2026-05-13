from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
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
    'training_dag',
    default_args=default_args,
    description='Train and evaluate models',
    schedule_interval='0 2 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['training'],
)


def train_model(model_type: str):
    """Train a specific model"""
    import os
    import yaml
    
    # Load params
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Update model type
    params['training']['model'] = model_type
    
    # Save updated params
    with open('params.yaml', 'w') as f:
        yaml.dump(params, f)
    
    # Run training
    from src.training.run_training import main
    main()


def train_rnn():
    train_model('rnn')


def train_lstm():
    train_model('lstm')


def train_gru():
    train_model('gru')


def evaluate_and_register():
    """Evaluate all models and register best"""
    # TODO: Implement model evaluation and registration logic
    pass


# Define tasks
train_rnn_task = PythonOperator(
    task_id='train_rnn',
    python_callable=train_rnn,
    dag=dag,
)

train_lstm_task = PythonOperator(
    task_id='train_lstm',
    python_callable=train_lstm,
    dag=dag,
)

train_gru_task = PythonOperator(
    task_id='train_gru',
    python_callable=train_gru,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_and_register',
    python_callable=evaluate_and_register,
    dag=dag,
)

# Set dependencies: all training tasks run in parallel, then evaluate
[train_rnn_task, train_lstm_task, train_gru_task] >> evaluate_task
