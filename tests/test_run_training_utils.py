"""
Tests for run_training utility functions and ModelRegistry checkpoint loading.
These tests cover the previously-uncovered src/training/run_training.py and
src/api/model_loader.py code paths without requiring a live MLflow server.
"""

from __future__ import annotations

import torch
import yaml

from src.models.lstm_model import LSTMModel
from src.training.run_training import (
    load_batch_size,
    load_feature_arrays,
    register_best_model,
    resolve_data_version,
    resolve_device,
)

# ── resolve_device ────────────────────────────────────────────────────────────


def test_resolve_device_cpu_explicit() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_explicit() -> None:
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_returns_string() -> None:
    result = resolve_device("auto")
    assert result in ("cpu", "cuda")


# ── load_batch_size ───────────────────────────────────────────────────────────


def test_load_batch_size_reads_params_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    params = {"training": {"batch_size": 32}}
    (tmp_path / "params.yaml").write_text(yaml.dump(params))
    assert load_batch_size() == 32


def test_load_batch_size_returns_int(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    params = {"training": {"batch_size": "16"}}
    (tmp_path / "params.yaml").write_text(yaml.dump(params))
    result = load_batch_size()
    assert isinstance(result, int)
    assert result == 16


# ── resolve_data_version ──────────────────────────────────────────────────────


def test_resolve_data_version_returns_string() -> None:
    # dvc may not be configured in CI — either way we get a non-empty string
    result = resolve_data_version()
    assert isinstance(result, str)
    assert len(result) > 0


def test_resolve_data_version_fallback_when_dvc_fails(monkeypatch) -> None:
    from subprocess import CompletedProcess
    monkeypatch.setattr(
        "src.training.run_training.run",
        lambda *a, **kw: CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
    )
    assert resolve_data_version() == "unknown"


# ── register_best_model (error path) ─────────────────────────────────────────


def test_register_best_model_silently_swallows_exceptions() -> None:
    # Should not raise even if MLflow is unreachable
    register_best_model("fake-run-id", "direction", {"accuracy": 0.7})


# ── load_feature_arrays ───────────────────────────────────────────────────────


def test_load_feature_arrays_returns_correct_keys(tmp_path, monkeypatch) -> None:
    import numpy as np

    monkeypatch.chdir(tmp_path)
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)

    # Write minimal fake feature arrays
    X = np.zeros((10, 5, 4), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    for split in ("train", "val", "test"):
        np.save(features_dir / f"X_{split}.npy", X)
        np.save(features_dir / f"y_direction_{split}.npy", y)

    params = {"training": {"batch_size": 4}}
    (tmp_path / "params.yaml").write_text(yaml.dump(params))

    arrays = load_feature_arrays(features_dir, "direction")

    assert "X_train" in arrays
    assert "train_loader" in arrays
    assert "val_loader" in arrays
    assert "test_loader" in arrays


# ── ModelRegistry._load_from_checkpoint ──────────────────────────────────────


def test_model_registry_returns_none_when_no_files(tmp_path) -> None:
    import os
    os.environ["MODEL_REGISTRY_PATH"] = str(tmp_path)
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:9999"

    from src.api.model_loader import ModelRegistry
    registry = ModelRegistry()
    assert registry.get_model("direction") is None


def test_model_registry_loads_from_checkpoint(tmp_path) -> None:
    import os

    # Save a real LSTM state dict to the tmp models dir
    model = LSTMModel(input_size=10, hidden_size=8, num_layers=1, output_size=3, dropout=0.0)
    checkpoint_path = tmp_path / "lstm_direction_best.pt"
    torch.save(model.state_dict(), checkpoint_path)

    os.environ["MODEL_REGISTRY_PATH"] = str(tmp_path)
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:9999"

    from src.api.model_loader import ModelRegistry
    registry = ModelRegistry()
    loaded = registry.get_model("direction")

    assert loaded is not None
    assert isinstance(loaded, torch.nn.Module)

    # Verify it actually runs inference
    loaded.eval()
    with torch.no_grad():
        out = loaded(torch.randn(1, 5, 10))
    assert out.shape == (1, 3)


def test_model_registry_models_loaded_flag(tmp_path) -> None:
    import os

    model = LSTMModel(input_size=10, hidden_size=8, num_layers=1, output_size=1, dropout=0.0)
    torch.save(model.state_dict(), tmp_path / "lstm_return_best.pt")

    os.environ["MODEL_REGISTRY_PATH"] = str(tmp_path)
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:9999"

    from src.api.model_loader import ModelRegistry
    registry = ModelRegistry()
    assert not registry.models_loaded()
    registry.get_model("return")
    assert registry.models_loaded()
