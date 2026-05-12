import pytest

# Skip all tests in this module - Phase 4 tests not ready for CI yet
pytestmark = pytest.mark.skip(reason="Phase 4 training tests - will be enabled when Phase 4 is complete")

import mlflow
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.models.rnn_model import SimpleRNNModel
    from src.training.evaluator import Evaluator
    from src.training.trainer import Trainer
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


def test_trainer_runs_one_epoch(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    train_loader, val_loader = make_direction_dataloaders()
    model = SimpleRNNModel(10, 8, 1, 3, 0.0)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("phase4-trainer-one-epoch")

    with mlflow.start_run():
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            learning_rate=0.001,
            device="cpu",
            patience=2,
            task_name="direction",
        )
        trainer.train(epochs=1, model_name="rnn", params={"training": {"epochs": 1}})

    assert (tmp_path / "models" / "rnn_direction_best.pt").exists()


def test_early_stopping_triggers(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    train_loader, val_loader = make_direction_dataloaders()
    model = SimpleRNNModel(10, 8, 1, 3, 0.0)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("phase4-early-stop")

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=0.001,
        device="cpu",
        patience=2,
        task_name="direction",
    )

    call_count = {"value": 0}

    def fake_train_epoch(*args, **kwargs) -> float:
        return 1.0

    def fake_validate_epoch(*args, **kwargs) -> dict:
        call_count["value"] += 1
        return {"val_loss": 1.0, "val_accuracy": 0.5, "val_f1": 0.5}

    monkeypatch.setattr(trainer, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "_validate_epoch", fake_validate_epoch)

    with mlflow.start_run():
        history = trainer.train(epochs=10, model_name="rnn", params={"training": {"epochs": 10}})

    assert len(history["val_loss"]) < 10
    assert call_count["value"] == len(history["val_loss"])


def test_evaluator_returns_all_metrics() -> None:
    inputs = torch.randn(4, 24, 10)
    targets = torch.tensor([0, 1, 2, 1])
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    class StaticModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.outputs = [
                torch.tensor([[4.0, 1.0, 0.0], [0.0, 4.0, 1.0]]),
                torch.tensor([[0.0, 1.0, 4.0], [0.0, 4.0, 1.0]]),
            ]

        def forward(self, x):
            return self.outputs.pop(0)

    metrics = Evaluator(StaticModel(), dataloader, "direction", device="cpu").evaluate()

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics


def test_mlflow_run_logged(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "phase4-mlflow"
    mlflow.set_experiment(experiment_name)

    train_loader, val_loader = make_direction_dataloaders()
    model = SimpleRNNModel(10, 8, 1, 3, 0.0)
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=0.001,
        device="cpu",
        patience=2,
        task_name="direction",
    )

    with mlflow.start_run():
        trainer.train(
            epochs=1,
            model_name="rnn",
            params={"training": {"epochs": 1, "batch_size": 4}, "model": "rnn"},
        )

    runs = mlflow.search_runs(experiment_names=[experiment_name])

    assert not runs.empty
    assert "metrics.train_loss" in runs.columns
    assert "params.training.epochs" in runs.columns


def make_direction_dataloaders() -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(7)
    X_train = torch.randn(8, 24, 10, generator=generator)
    y_train = torch.randint(0, 3, (8,), generator=generator)
    X_val = torch.randn(4, 24, 10, generator=generator)
    y_val = torch.randint(0, 3, (4,), generator=generator)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=2)
    return train_loader, val_loader
