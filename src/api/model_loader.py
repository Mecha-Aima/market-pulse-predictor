import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
import torch
from mlflow.tracking import MlflowClient

from src.models.gru_model import GRUModel
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import SimpleRNNModel

_ARCH_CLASSES = {"lstm": LSTMModel, "gru": GRUModel, "rnn": SimpleRNNModel}

_TASK_OUTPUT_SIZES = {"direction": 3, "return": 1, "volatility": 1}

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages loading and caching of trained models"""

    def __init__(self):
        self.models = {}
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.model_path = Path(os.getenv("MODEL_REGISTRY_PATH", "./models/"))
        self.client = None

    def _init_mlflow_client(self):
        """Initialize MLflow client with error handling"""
        if self.client is None:
            try:
                mlflow.set_tracking_uri(self.mlflow_uri)
                self.client = MlflowClient()
                logger.info(f"Connected to MLflow at {self.mlflow_uri}")
            except Exception as e:
                logger.warning(f"Failed to connect to MLflow: {e}")
                self.client = None

    def get_model(self, task: str) -> Optional[torch.nn.Module]:
        """Get model for a specific task (direction, return, volatility)"""
        if task in self.models:
            return self.models[task]

        # Try loading from MLflow first
        model = self._load_from_mlflow(task)
        if model is not None:
            self.models[task] = model
            return model

        # Fallback to local .pt files
        model = self._load_from_checkpoint(task)
        if model is not None:
            self.models[task] = model
            return model

        logger.warning(f"No model found for task: {task}")
        return None

    def _load_from_mlflow(self, task: str) -> Optional[torch.nn.Module]:
        """Load production model from MLflow registry"""
        self._init_mlflow_client()
        if self.client is None:
            return None

        try:
            model_name = f"MarketPulsePredictor-{task}"
            # Get production version
            versions = self.client.search_model_versions(f"name='{model_name}'")
            prod_versions = [v for v in versions if v.current_stage == "Production"]

            if not prod_versions:
                logger.info(f"No production model found for {model_name}")
                return None

            version = prod_versions[0]
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded {model_name} from MLflow (version {version.version})")
            return model

        except Exception as e:
            logger.warning(f"Failed to load model from MLflow: {e}")
            return None

    def _load_from_checkpoint(self, task: str) -> Optional[torch.nn.Module]:
        """Load model from local .pt checkpoint"""
        for arch_name in ("lstm", "gru", "rnn"):
            checkpoint_path = self.model_path / f"{arch_name}_{task}_best.pt"
            if checkpoint_path.exists():
                try:
                    state_dict = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=True
                    )
                    model = self._reconstruct_from_state_dict(
                        state_dict, arch_name, task
                    )
                    logger.info(f"Loaded {arch_name} model for task={task} from {checkpoint_path}")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load {checkpoint_path}: {e}")

        return None

    @staticmethod
    def _reconstruct_from_state_dict(
        state_dict: dict, arch_name: str, task: str
    ) -> torch.nn.Module:
        """Reconstruct a model instance from a saved state dict."""
        # Infer architecture dimensions from the state dict itself.
        hh_keys = sorted(k for k in state_dict if "weight_hh_l" in k)
        hidden_size = state_dict[hh_keys[0]].shape[1]
        num_layers = len(hh_keys)
        ih_key = next(k for k in state_dict if "weight_ih_l0" in k)
        input_size = state_dict[ih_key].shape[1]
        output_size = _TASK_OUTPUT_SIZES[task]

        ModelClass = _ARCH_CLASSES[arch_name]
        model = ModelClass(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=0.0,
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_latest_features(self, ticker: str) -> Optional[dict]:
        """Read most recent features for a ticker"""
        # This will be implemented when we have actual feature data
        # For now, return None
        return None

    def models_loaded(self) -> bool:
        """Check if any models are loaded"""
        return len(self.models) > 0


# Global registry instance
registry = ModelRegistry()
