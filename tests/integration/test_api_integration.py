"""
Integration tests for API with real model predictions.

Tests API endpoints with actual model inference.
"""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.main import app

try:
    from src.models.rnn_model import SimpleRNNModel

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Skip until Phase 4 is wired in CI
pytestmark = pytest.mark.skip(
    reason="Phase 4 integration tests - will be enabled when Phase 4 is complete"
)


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API with real models."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_tickers_endpoint(self, client):
        response = client.get("/tickers")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_results_endpoint(self, client):
        response = client.get("/results")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@pytest.mark.integration
class TestAPIPredictionIntegration:
    """Integration tests for model inference."""

    def test_model_inference(self, fixture_features_data):
        features_df = fixture_features_data.copy()
        if len(features_df) < 5:
            pytest.skip("Not enough data")

        cols = [c for c in ["close", "volume"] if c in features_df.columns]
        if not cols:
            pytest.skip("No numeric features available")

        X = torch.tensor(features_df[cols].head(5).values, dtype=torch.float32).unsqueeze(1)

        model = SimpleRNNModel(len(cols), 16, 1, 2, 0.0)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X), dim=1)
            preds = torch.argmax(probs, dim=1)

        assert preds.shape[0] == 5
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_prediction_consistency(self, fixture_features_data):
        features_df = fixture_features_data.copy()
        if len(features_df) < 5:
            pytest.skip("Not enough data")

        cols = [c for c in ["close", "volume"] if c in features_df.columns]
        if not cols:
            pytest.skip("No numeric features available")

        X = torch.tensor(features_df[cols].head(5).values, dtype=torch.float32).unsqueeze(1)
        model = SimpleRNNModel(len(cols), 16, 1, 2, 0.0)
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(X), model(X), atol=1e-6)


@pytest.mark.integration
class TestAPIPipeline:
    """Integration tests for complete API pipeline."""

    def test_end_to_end_api_flow(self, fixture_features_data, tmp_path):
        features_df = fixture_features_data.copy()
        if len(features_df) < 10:
            pytest.skip("Not enough data")

        cols = [c for c in ["close", "volume"] if c in features_df.columns]
        if not cols:
            pytest.skip("No numeric features available")

        split = int(len(features_df) * 0.8)
        train_df = features_df[:split]
        test_df = features_df[split:]

        X_train = torch.tensor(train_df[cols].values, dtype=torch.float32).unsqueeze(1)
        labels = (
            train_df["label"].values
            if "label" in train_df.columns
            else np.random.randint(0, 2, len(train_df))
        )
        y_train = torch.tensor(labels, dtype=torch.long)

        model = SimpleRNNModel(len(cols), 16, 1, 2, 0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        criterion(model(X_train), y_train).backward()
        optimizer.step()

        model_path = tmp_path / "api_model.pt"
        torch.save(model.state_dict(), model_path)

        loaded = SimpleRNNModel(len(cols), 16, 1, 2, 0.0)
        loaded.load_state_dict(torch.load(model_path))

        X_test = torch.tensor(test_df[cols].values, dtype=torch.float32).unsqueeze(1)
        loaded.eval()
        with torch.no_grad():
            probs = torch.softmax(loaded(X_test), dim=1)
            preds = torch.argmax(probs, dim=1)

        assert model_path.exists()
        assert preds.shape[0] == len(test_df)

        results = [
            {
                "index": i,
                "prediction": p.item(),
                "probability_up": prob[1].item(),
                "probability_down": prob[0].item(),
            }
            for i, (p, prob) in enumerate(zip(preds, probs, strict=True))
        ]

        assert len(results) == len(test_df)
        assert all(0 <= r["probability_up"] <= 1 for r in results)
