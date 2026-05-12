from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from subprocess import run

import mlflow
import numpy as np
import torch
import yaml
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader, TensorDataset

from src.models.gru_model import GRUModel
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import SimpleRNNModel
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer

MODEL_FACTORIES = {
    "rnn": SimpleRNNModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
}


def run_training_pipeline() -> dict:
    params = yaml.safe_load((Path.cwd() / "params.yaml").read_text())
    training_params = params["training"]
    task_name = training_params["task"]
    model_name = training_params["model"]
    device = resolve_device(training_params["device"])
    arrays = load_feature_arrays(Path.cwd() / "data" / "features", task_name)
    input_size = arrays["X_train"].shape[-1]
    output_size = 3 if task_name == "direction" else 1
    model = MODEL_FACTORIES[model_name](
        input_size=input_size,
        hidden_size=training_params["hidden_size"],
        num_layers=training_params["num_layers"],
        output_size=output_size,
        dropout=training_params["dropout"],
    )

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "file://" + str(Path.cwd() / "mlruns"))
    )
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "market_pulse")
    mlflow.set_experiment(experiment_name)
    run_name = f"{model_name}_{task_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
    tags = {
        "model_type": model_name,
        "task": task_name,
        "data_version": resolve_data_version(),
    }

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tags(tags)
        trainer = Trainer(
            model=model,
            train_dataloader=arrays["train_loader"],
            val_dataloader=arrays["val_loader"],
            learning_rate=training_params["learning_rate"],
            device=device,
            patience=training_params["early_stopping_patience"],
            task_name=task_name,
        )
        history = trainer.train(
            epochs=training_params["epochs"],
            model_name=model_name,
            params=params,
        )
        evaluator = Evaluator(model, arrays["test_loader"], task_name, device=device)
        metrics = evaluator.evaluate()
        register_best_model(active_run.info.run_id, task_name, metrics)
        return {"history": history, "metrics": metrics}


def load_feature_arrays(features_dir: Path, task_name: str) -> dict:
    X_train = np.load(features_dir / "X_train.npy")
    X_val = np.load(features_dir / "X_val.npy")
    X_test = np.load(features_dir / "X_test.npy")
    target_prefix = {
        "direction": "y_direction",
        "return": "y_return",
        "volatility": "y_volatility",
    }[task_name]
    y_train = np.load(features_dir / f"{target_prefix}_train.npy")
    y_val = np.load(features_dir / f"{target_prefix}_val.npy")
    y_test = np.load(features_dir / f"{target_prefix}_test.npy")
    return {
        "X_train": X_train,
        "train_loader": DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)),
            batch_size=load_batch_size(),
        ),
        "val_loader": DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val)),
            batch_size=load_batch_size(),
        ),
        "test_loader": DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)),
            batch_size=load_batch_size(),
        ),
    }


def load_batch_size() -> int:
    params = yaml.safe_load((Path.cwd() / "params.yaml").read_text())
    return int(params["training"]["batch_size"])


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def resolve_data_version() -> str:
    result = run(
        ["dvc", "data", "status", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return "unknown"


def register_best_model(run_id: str, task_name: str, metrics: dict) -> None:
    model_uri = f"runs:/{run_id}/model"
    registered_name = f"MarketPulsePredictor-{task_name}"
    try:
        model_version = mlflow.register_model(model_uri=model_uri, name=registered_name)
        client = MlflowClient()
        if hasattr(client, "transition_model_version_stage"):
            client.transition_model_version_stage(
                name=registered_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True,
            )
    except Exception:
        return


def main() -> None:
    run_training_pipeline()


if __name__ == "__main__":
    main()
