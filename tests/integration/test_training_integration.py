"""
Integration tests for model training pipeline.

Tests model training with real data (1 epoch for speed).
"""

import numpy as np
import pytest
import torch

try:
    from src.models.gru_model import GRUModel
    from src.models.lstm_model import LSTMModel
    from src.models.rnn_model import SimpleRNNModel

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Skip until Phase 4 is wired in CI
pytestmark = pytest.mark.skip(
    reason="Phase 4 integration tests - will be enabled when Phase 4 is complete"
)


def _make_tensors(features_df, feature_cols, n_classes=2):
    available = [c for c in feature_cols if c in features_df.columns]
    if not available:
        return None, None, available
    X = torch.tensor(
        features_df[available].values, dtype=torch.float32
    ).unsqueeze(1)
    labels = (
        features_df["label"].values
        if "label" in features_df.columns
        else np.random.randint(0, n_classes, len(features_df))
    )
    y = torch.tensor(labels, dtype=torch.long)
    return X, y, available


@pytest.mark.integration
class TestModelTrainingIntegration:
    """Integration tests for model training with real data."""

    def test_train_rnn_one_epoch(self, fixture_features_data, tmp_path):
        features_df = fixture_features_data.copy()
        split = int(len(features_df) * 0.8)
        train_df = features_df[:split]
        if len(train_df) < 10:
            pytest.skip("Not enough data for training")

        X, y, cols = _make_tensors(train_df, ["close", "volume", "returns"])
        if X is None:
            pytest.skip("No numeric features available")

        model = SimpleRNNModel(len(cols), 32, 1, 2, 0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        assert isinstance(loss.item(), float)
        assert loss.item() > 0
        assert not np.isnan(loss.item())

    def test_train_lstm_one_epoch(self, fixture_features_data):
        features_df = fixture_features_data.copy()
        train_df = features_df[: int(len(features_df) * 0.8)]
        if len(train_df) < 10:
            pytest.skip("Not enough data for training")

        X, y, cols = _make_tensors(train_df, ["close", "volume"])
        if X is None:
            pytest.skip("No numeric features available")

        model = LSTMModel(len(cols), 32, 1, 2, 0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        assert not np.isnan(loss.item())

    def test_train_gru_one_epoch(self, fixture_features_data):
        features_df = fixture_features_data.copy()
        train_df = features_df[: int(len(features_df) * 0.8)]
        if len(train_df) < 10:
            pytest.skip("Not enough data for training")

        X, y, cols = _make_tensors(train_df, ["close", "volume"])
        if X is None:
            pytest.skip("No numeric features available")

        model = GRUModel(len(cols), 32, 1, 2, 0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        assert not np.isnan(loss.item())


@pytest.mark.integration
class TestModelEvaluationIntegration:
    """Integration tests for model evaluation."""

    def test_evaluate_trained_model(self, fixture_features_data):
        features_df = fixture_features_data.copy()
        split = int(len(features_df) * 0.8)
        train_df = features_df[:split]
        test_df = features_df[split:]

        if len(train_df) < 10 or len(test_df) < 5:
            pytest.skip("Not enough data for evaluation")

        X_train, y_train, cols = _make_tensors(train_df, ["close", "volume"])
        if X_train is None:
            pytest.skip("No numeric features available")

        X_test = torch.tensor(test_df[cols].values, dtype=torch.float32).unsqueeze(1)
        labels = (
            test_df["label"].values
            if "label" in test_df.columns
            else np.random.randint(0, 2, len(test_df))
        )
        y_test = torch.tensor(labels, dtype=torch.long)

        model = SimpleRNNModel(len(cols), 32, 1, 2, 0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        criterion(model(X_train), y_train).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test)
            preds = torch.argmax(model(X_test), dim=1)
            acc = (preds == y_test).float().mean()

        assert test_loss.item() >= 0
        assert 0 <= acc.item() <= 1

    def test_model_prediction_consistency(self, fixture_features_data):
        features_df = fixture_features_data.copy()
        if len(features_df) < 5:
            pytest.skip("Not enough data")

        X, _, cols = _make_tensors(features_df.head(5), ["close", "volume"])
        if X is None:
            pytest.skip("No numeric features available")

        model = SimpleRNNModel(len(cols), 32, 1, 2, 0.0)
        model.eval()
        with torch.no_grad():
            out1 = model(X)
            out2 = model(X)

        assert torch.allclose(out1, out2, atol=1e-6)


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for complete training pipeline."""

    def test_end_to_end_training_pipeline(self, fixture_features_data, tmp_path):
        features_df = fixture_features_data.copy()
        split = int(len(features_df) * 0.8)
        train_df = features_df[:split]
        test_df = features_df[split:]

        if len(train_df) < 10 or len(test_df) < 5:
            pytest.skip("Not enough data")

        X_train, y_train, cols = _make_tensors(train_df, ["close", "volume"])
        if X_train is None:
            pytest.skip("No numeric features available")

        X_test = torch.tensor(test_df[cols].values, dtype=torch.float32).unsqueeze(1)

        model = SimpleRNNModel(len(cols), 32, 1, 2, 0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        train_loss = criterion(model(X_train), y_train)
        train_loss.backward()
        optimizer.step()

        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)
        assert model_path.exists()

        loaded = SimpleRNNModel(len(cols), 32, 1, 2, 0.0)
        loaded.load_state_dict(torch.load(model_path))
        loaded.eval()
        with torch.no_grad():
            preds = torch.argmax(loaded(X_test), dim=1)

        assert preds.shape[0] == len(test_df)
        assert train_loss.item() > 0
