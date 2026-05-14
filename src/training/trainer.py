from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from torch import nn


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        learning_rate: float,
        device: str,
        patience: int,
        task_name: str,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.device = device
        self.patience = patience
        self.task_name = task_name
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = self._build_loss()

    def train(
        self, epochs: int, model_name: str, params: dict | None = None
    ) -> dict[str, list[float]]:
        history: dict[str, list[float]] = defaultdict(list)
        if params:
            mlflow.log_params(self._flatten_params(params))

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        checkpoint_path = self._checkpoint_path(model_name)

        for epoch in range(epochs):
            train_loss = self._train_epoch()
            validation = self._validate_epoch()
            history["train_loss"].append(train_loss)
            for key, value in validation.items():
                history[key].append(value)

            metrics = {"train_loss": train_loss, **validation}
            mlflow.log_metrics(metrics, step=epoch)

            if validation["val_loss"] < best_loss:
                best_loss = validation["val_loss"]
                best_state = deepcopy(self.model.state_dict())
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        try:
            from mlflow.models.signature import infer_signature
            
            # Generate input example and signature
            example_features, _ = next(iter(self.train_dataloader))
            example_input = example_features[:1].to(self.device)
            example_output = self.model(example_input)
            
            input_example = example_input.cpu().numpy()
            signature = infer_signature(input_example, example_output.detach().cpu().numpy())
            
            # Clean local version from torch requirement to avoid MLflow warning
            reqs = mlflow.pytorch.get_default_pip_requirements()
            cleaned_reqs = [r.split("+")[0] if r.startswith("torch") else r for r in reqs]
            
            mlflow.pytorch.log_model(
                self.model, 
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                pip_requirements=cleaned_reqs
            )
        except Exception as e:
            print(f"Warning: Failed to log model signature: {e}")
            mlflow.pytorch.log_model(self.model, artifact_path="model")

        return dict(history)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        for features, targets in self.train_dataloader:
            features = features.to(self.device)
            prepared_targets = self._prepare_targets(targets)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.loss_fn(outputs, prepared_targets)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
        return total_loss / max(batch_count, 1)

    def _validate_epoch(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        predictions = []
        targets = []
        with torch.no_grad():
            for features, batch_targets in self.val_dataloader:
                features = features.to(self.device)
                prepared_targets = self._prepare_targets(batch_targets)
                outputs = self.model(features)
                loss = self.loss_fn(outputs, prepared_targets)
                total_loss += float(loss.item())
                batch_count += 1
                predictions.append(outputs.detach().cpu())
                targets.append(batch_targets.detach().cpu())

        metrics = {"val_loss": total_loss / max(batch_count, 1)}
        if not predictions:
            return metrics

        prediction_tensor = torch.cat(predictions)
        target_tensor = torch.cat(targets)
        if self.task_name == "return":
            predicted_values = prediction_tensor.squeeze(-1).numpy()
            actual_values = target_tensor.numpy()
            metrics["val_rmse"] = float(mean_squared_error(actual_values, predicted_values) ** 0.5)
        else:
            predicted_classes = self._classification_outputs(prediction_tensor)
            normalized_targets = self._normalize_classification_targets(target_tensor)
            metrics["val_accuracy"] = float(accuracy_score(normalized_targets, predicted_classes))
            metrics["val_f1"] = float(
                f1_score(normalized_targets, predicted_classes, average="macro")
            )
        return metrics

    def _get_pos_weight(self) -> torch.Tensor | None:
        if self.task_name != "volatility":
            return None
        total = 0
        pos = 0
        for _, targets in self.train_dataloader:
            total += targets.numel()
            pos += targets.sum().item()
        if pos == 0:
            return torch.tensor([1.0], device=self.device)
        return torch.tensor([(total - pos) / pos], device=self.device)

    def _build_loss(self):
        if self.task_name == "direction":
            return nn.CrossEntropyLoss()
        if self.task_name == "return":
            return nn.MSELoss()
        pos_weight = self._get_pos_weight()
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(self.device)
        if self.task_name == "direction":
            if torch.min(targets).item() < 0:
                targets = targets + 1
            return targets.long()
        if self.task_name == "return":
            return targets.float().view(-1, 1)
        return targets.float().view(-1, 1)

    def _classification_outputs(self, predictions: torch.Tensor):
        if self.task_name == "volatility":
            return (torch.sigmoid(predictions.squeeze(-1)) >= 0.5).int().numpy()
        return predictions.argmax(dim=1).numpy()

    @staticmethod
    def _normalize_classification_targets(targets: torch.Tensor):
        if torch.min(targets).item() < 0:
            return (targets + 1).numpy()
        return targets.numpy()

    @staticmethod
    def _flatten_params(params: dict, prefix: str = "") -> dict[str, str | int | float | bool]:
        flattened = {}
        for key, value in params.items():
            compound_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(Trainer._flatten_params(value, compound_key))
            else:
                flattened[compound_key] = value
        return flattened

    def _checkpoint_path(self, model_name: str) -> Path:
        return Path.cwd() / "models" / f"{model_name}_{self.task_name}_best.pt"
