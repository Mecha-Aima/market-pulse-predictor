"""
Deployment readiness tests.

Model-dependent tests are skipped when the models/ directory is absent (CI environment).
All other tests run unconditionally and validate real project artifacts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ── Helpers ────────────────────────────────────────────────────────────────────

MODEL_ARCH_MAP = {
    "gru": "src.models.gru_model.GRUModel",
    "lstm": "src.models.lstm_model.LSTMModel",
    "rnn": "src.models.rnn_model.SimpleRNNModel",
}

# All 9 expected model files with their known architecture properties.
# input_size=28 matches feature_columns.json produced by TimeSeriesBuilder;
# sequence_length=10 is set in params.yaml.
EXPECTED_MODELS = [
    ("gru_direction_best.pt", "gru", 28, 3),
    ("gru_return_best.pt", "gru", 28, 1),
    ("gru_volatility_best.pt", "gru", 28, 1),
    ("lstm_direction_best.pt", "lstm", 28, 3),
    ("lstm_return_best.pt", "lstm", 28, 1),
    ("lstm_volatility_best.pt", "lstm", 28, 1),
    ("rnn_direction_best.pt", "rnn", 28, 3),
    ("rnn_return_best.pt", "rnn", 28, 1),
    ("rnn_volatility_best.pt", "rnn", 28, 1),
]

models_present = MODELS_DIR.exists() and any(MODELS_DIR.glob("*.pt"))
skip_without_models = pytest.mark.skipif(
    not models_present,
    reason="models/ directory not present — run `dvc pull` to fetch trained models",
)


def _load_model_class(arch: str):
    """Import and return the model class for a given architecture name."""
    import importlib

    module_path, class_name = MODEL_ARCH_MAP[arch].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ── Tests ─────────────────────────────────────────────────────────────────────


@skip_without_models
@pytest.mark.parametrize("filename,arch,input_size,output_size", EXPECTED_MODELS)
def test_model_file_loads_and_runs_forward_pass(
    filename: str, arch: str, input_size: int, output_size: int
) -> None:
    """Each .pt state dict loads cleanly and produces the expected output shape."""
    pt_path = MODELS_DIR / filename
    assert pt_path.exists(), f"Missing model file: {pt_path}"

    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    assert isinstance(state_dict, dict), f"{filename} should be a state dict"

    # Infer hidden_size and num_layers directly from the state dict so the test
    # is robust to hyperparameter changes between training runs.
    hh_keys = sorted(k for k in state_dict if "weight_hh_l" in k)
    assert hh_keys, f"{filename}: no weight_hh_l* keys found in state dict"
    hidden_size = state_dict[hh_keys[0]].shape[1]
    num_layers = len(hh_keys)

    ModelClass = _load_model_class(arch)
    model = ModelClass(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=0.0,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # sequence_length=10 matches params.yaml; batch_size=1 for unit test.
    x = torch.randn(1, 10, input_size)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, output_size), (
        f"{filename}: expected output shape (1, {output_size}), got {out.shape}"
    )
    assert not torch.isnan(out).any(), f"{filename}: forward pass produced NaN"


def test_docker_compose_prod_config_valid() -> None:
    """docker-compose.prod.yml passes `docker compose config` validation."""
    compose_file = PROJECT_ROOT / "docker-compose.prod.yml"
    assert compose_file.exists(), "docker-compose.prod.yml does not exist"

    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "config", "--quiet"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"docker-compose.prod.yml is invalid:\n{result.stderr}"
    )


def test_env_example_has_all_required_keys() -> None:
    """
    .env.example documents every key the application reads at runtime.
    This ensures a new deployment doesn't silently omit required configuration.
    """
    required_keys = {
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "DVC_REMOTE_BUCKET",
        "AIRFLOW__CORE__EXECUTOR",
        "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
        "API_HOST",
        "API_PORT",
        "MODEL_REGISTRY_PATH",
        "TARGET_TICKERS",
    }

    env_example = PROJECT_ROOT / ".env.example"
    assert env_example.exists(), ".env.example file is missing"

    present_keys: set[str] = set()
    for line in env_example.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            present_keys.add(key)

    missing = required_keys - present_keys
    assert not missing, f".env.example is missing required keys: {sorted(missing)}"


def test_dvc_remote_configured_as_s3() -> None:
    """DVC is configured with an S3 remote — required for data versioning on EC2."""
    dvc_config_path = PROJECT_ROOT / ".dvc" / "config"
    assert dvc_config_path.exists(), ".dvc/config not found — run `dvc init`"

    content = dvc_config_path.read_text()

    # Verify a default remote is set in [core]
    assert "remote = " in content, ".dvc/config missing core.remote setting"

    # Extract the remote name and verify its URL points to S3.
    # DVC uses single-quoted section headers: ['remote "name"'] which configparser
    # does not handle, so we parse with a simple regex instead.
    import re

    remote_match = re.search(r"remote\s*=\s*(\S+)", content)
    assert remote_match, "Could not parse remote name from .dvc/config"
    remote_name = remote_match.group(1)

    url_match = re.search(rf'\[\'remote "{re.escape(remote_name)}"\'\].*?url\s*=\s*(\S+)', content, re.DOTALL)
    assert url_match, f"Could not find url for remote '{remote_name}' in .dvc/config"
    url = url_match.group(1)

    assert url.startswith("s3://"), (
        f"DVC remote URL should start with s3://, got: {url!r}"
    )


@skip_without_models
def test_model_registry_loads_real_pt_files() -> None:
    """
    ModelRegistry.get_model() returns a working nn.Module when .pt files are present,
    without requiring a live MLflow server.
    """
    import os

    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:9999"  # deliberately unreachable
    os.environ["MODEL_REGISTRY_PATH"] = str(MODELS_DIR)

    # Re-import to pick up the patched env vars
    import importlib

    import src.api.model_loader as loader_module

    importlib.reload(loader_module)

    registry = loader_module.ModelRegistry()
    model = registry.get_model("direction")

    assert model is not None, (
        "ModelRegistry.get_model('direction') returned None — "
        "check that lstm_direction_best.pt (or gru/rnn) exists in models/"
    )
    assert isinstance(model, torch.nn.Module), (
        f"Expected nn.Module, got {type(model)}"
    )

    # Verify the loaded model produces valid output.
    # Use input_size=28 (matches feature columns) and seq_len=10 (params.yaml).
    model.eval()
    x = torch.randn(1, 10, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape[0] == 1, f"Unexpected output batch dimension: {out.shape}"
    assert not torch.isnan(out).any(), "Loaded model produces NaN on forward pass"
