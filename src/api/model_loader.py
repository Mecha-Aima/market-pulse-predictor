import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
import torch
from mlflow.tracking import MlflowClient

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
        # Look for any model checkpoint for this task
        patterns = [
            f"lstm_{task}_best.pt",
            f"gru_{task}_best.pt",
            f"rnn_{task}_best.pt",
        ]

        for pattern in patterns:
            checkpoint_path = self.model_path / pattern
            if checkpoint_path.exists():
                try:
                    model = torch.load(checkpoint_path, map_location="cpu")
                    logger.info(f"Loaded model from {checkpoint_path}")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load {checkpoint_path}: {e}")

        return None

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
