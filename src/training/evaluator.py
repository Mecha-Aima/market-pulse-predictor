from __future__ import annotations

import math

import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


class Evaluator:
    def __init__(self, model, test_dataloader, task_name: str, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.test_dataloader = test_dataloader
        self.task_name = task_name
        self.device = device

    def evaluate(self) -> dict:
        self.model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for features, batch_targets in self.test_dataloader:
                features = features.to(self.device)
                outputs = self.model(features)
                predictions.append(outputs.detach().cpu())
                targets.append(batch_targets.detach().cpu())

        prediction_tensor = torch.cat(predictions) if predictions else torch.empty(0)
        target_tensor = torch.cat(targets) if targets else torch.empty(0)

        if self.task_name == "return":
            metrics = self._regression_metrics(prediction_tensor, target_tensor)
        else:
            metrics = self._classification_metrics(prediction_tensor, target_tensor)

        if mlflow.active_run():
            scalar_metrics = {
                key: value for key, value in metrics.items() if isinstance(value, (int, float))
            }
            mlflow.log_metrics(scalar_metrics)
        return metrics

    def _classification_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        predicted_classes = self._classification_outputs(predictions)
        normalized_targets = self._normalize_classification_targets(targets)
        return {
            "accuracy": float(accuracy_score(normalized_targets, predicted_classes)),
            "f1_score": float(f1_score(normalized_targets, predicted_classes, average="macro")),
            "confusion_matrix": confusion_matrix(normalized_targets, predicted_classes).tolist(),
            "classification_report": classification_report(
                normalized_targets,
                predicted_classes,
                output_dict=True,
                zero_division=0,
            ),
        }

    def _regression_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        predicted_values = predictions.squeeze(-1).numpy()
        actual_values = targets.numpy()
        return {
            "rmse": float(math.sqrt(mean_squared_error(actual_values, predicted_values))),
            "mae": float(mean_absolute_error(actual_values, predicted_values)),
        }

    def _classification_outputs(self, predictions: torch.Tensor) -> np.ndarray:
        if self.task_name == "volatility":
            return (torch.sigmoid(predictions.squeeze(-1)) >= 0.5).int().numpy()
        return predictions.argmax(dim=1).numpy()

    @staticmethod
    def _normalize_classification_targets(targets: torch.Tensor) -> np.ndarray:
        if targets.numel() == 0:
            return np.array([])
        if torch.min(targets).item() < 0:
            return (targets + 1).numpy()
        return targets.numpy()
