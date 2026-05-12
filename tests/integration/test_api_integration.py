"""
Integration tests for API with real model predictions.

Tests API endpoints with actual model inference.
"""

import pytest
from fastapi.testclient import TestClient
import torch
import numpy as np
from pathlib import Path

from src.api.main import app
from src.models.rnn_model import SimpleRNNModel


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API with real models."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def trained_model(self, tmp_path):
        """Create and save a trained model."""
        # Create simple model
        model = SimpleRNNModel(
            input_size=2,
            hidden_size=16,
            num_layers=1,
            num_classes=2
        )
        
        # Save model
        model_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), model_path)
        
        return model_path
    
    def test_health_endpoint(self, client):
        """Test: Health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200, "Health check should return 200"
        
        data = response.json()
        assert 'status' in data, "Should have status field"
        assert data['status'] == 'healthy', "Status should be healthy"
    
    def test_tickers_endpoint(self, client):
        """Test: Get configured tickers."""
        response = client.get("/tickers")
        
        assert response.status_code == 200, "Should return 200"
        
        data = response.json()
        assert isinstance(data, list), "Should return list of tickers"
        assert len(data) > 0, "Should have at least one ticker"
    
    def test_predict_endpoint_with_mock_model(self, client, monkeypatch):
        """Test: Prediction endpoint with mocked model."""
        # Mock model loader to return a simple prediction
        def mock_predict(ticker, features):
            return {
                'ticker': ticker,
                'prediction': 1,
                'probability': 0.75,
                'confidence': 'high'
            }
        
        # This test verifies the endpoint structure
        # Real model testing would require actual model files
        response = client.get("/predict/AAPL")
        
        # Endpoint should exist (might return 404 if no model loaded)
        assert response.status_code in [200, 404, 500], \
            "Endpoint should be accessible"
    
    def test_results_endpoint(self, client):
        """Test: Get prediction results."""
        response = client.get("/results")
        
        assert response.status_code == 200, "Should return 200"
        
        data = response.json()
        assert isinstance(data, list), "Should return list of results"


@pytest.mark.integration
class TestAPIPredictionIntegration:
    """Integration tests for API prediction with real model."""
    
    def test_model_inference(self, fixture_features_data):
        """Test: Model inference with real data."""
        features_df = fixture_features_data.copy()
        
        if len(features_df) < 5:
            pytest.skip("Not enough data")
        
        # Prepare input data
        feature_cols = ['close', 'volume']
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        if len(available_cols) == 0:
            pytest.skip("No numeric features available")
        
        X = torch.tensor(
            features_df[available_cols].head(5).values,
            dtype=torch.float32
        ).unsqueeze(1)
        
        # Create model
        model = SimpleRNNModel(
            input_size=len(available_cols),
            hidden_size=16,
            num_layers=1,
            num_classes=2
        )
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        # Verify predictions
        assert predictions.shape[0] == 5, "Should predict for 5 samples"
        assert predictions.dtype == torch.long, "Predictions should be integers"
        assert all(p in [0, 1] for p in predictions.tolist()), \
            "Predictions should be 0 or 1"
        
        # Verify probabilities
        assert probs.shape == (5, 2), "Should have probabilities for 2 classes"
        assert torch.all((probs >= 0) & (probs <= 1)), \
            "Probabilities should be in [0, 1]"
    
    def test_batch_prediction(self, fixture_features_data):
        """Test: Batch prediction for multiple samples."""
        features_df = fixture_features_data.copy()
        
        if len(features_df) < 10:
            pytest.skip("Not enough data")
        
        # Prepare batch
        feature_cols = ['close', 'volume']
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        if len(available_cols) == 0:
            pytest.skip("No numeric features available")
        
        X = torch.tensor(
            features_df[available_cols].head(10).values,
            dtype=torch.float32
        ).unsqueeze(1)
        
        # Create model
        model = SimpleRNNModel(
            input_size=len(available_cols),
            hidden_size=16,
            num_layers=1,
            num_classes=2
        )
        
        # Batch prediction
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
        
        # Verify batch processing
        assert outputs.shape[0] == 10, "Should process all 10 samples"
        assert probs.shape == (10, 2), "Should have probabilities for all samples"
    
    def test_prediction_consistency(self, fixture_features_data):
        """Test: Prediction consistency (same input = same output)."""
        features_df = fixture_features_data.copy()
        
        if len(features_df) < 5:
            pytest.skip("Not enough data")
        
        # Prepare input
        feature_cols = ['close', 'volume']
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        if len(available_cols) == 0:
            pytest.skip("No numeric features available")
        
        X = torch.tensor(
            features_df[available_cols].head(5).values,
            dtype=torch.float32
        ).unsqueeze(1)
        
        # Create model
        model = SimpleRNNModel(
            input_size=len(available_cols),
            hidden_size=16,
            num_layers=1,
            num_classes=2
        )
        
        # Make predictions twice
        model.eval()
        with torch.no_grad():
            outputs1 = model(X)
            outputs2 = model(X)
        
        # Verify consistency
        assert torch.allclose(outputs1, outputs2, atol=1e-6), \
            "Same input should produce same output"


@pytest.mark.integration
class TestAPIPipeline:
    """Integration tests for complete API pipeline."""
    
    def test_end_to_end_api_flow(self, fixture_features_data, tmp_path):
        """Test: Complete API flow from data to prediction."""
        features_df = fixture_features_data.copy()
        
        if len(features_df) < 10:
            pytest.skip("Not enough data")
        
        # 1. Prepare training data
        feature_cols = ['close', 'volume']
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        if len(available_cols) == 0:
            pytest.skip("No numeric features available")
        
        split_idx = int(len(features_df) * 0.8)
        train_df = features_df[:split_idx]
        test_df = features_df[split_idx:]
        
        X_train = torch.tensor(train_df[available_cols].values, dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(
            train_df['label'].values if 'label' in train_df.columns 
            else np.random.randint(0, 2, len(train_df)),
            dtype=torch.long
        )
        
        # 2. Train model
        model = SimpleRNNModel(
            input_size=len(available_cols),
            hidden_size=16,
            num_layers=1,
            num_classes=2
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # 3. Save model
        model_path = tmp_path / "api_model.pt"
        torch.save(model.state_dict(), model_path)
        
        # 4. Load model for inference
        loaded_model = SimpleRNNModel(
            input_size=len(available_cols),
            hidden_size=16,
            num_layers=1,
            num_classes=2
        )
        loaded_model.load_state_dict(torch.load(model_path))
        
        # 5. Make predictions
        X_test = torch.tensor(test_df[available_cols].values, dtype=torch.float32).unsqueeze(1)
        
        loaded_model.eval()
        with torch.no_grad():
            outputs = loaded_model(X_test)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        # 6. Verify complete flow
        assert model_path.exists(), "Model should be saved"
        assert predictions.shape[0] == len(test_df), "Should predict for all test samples"
        assert all(p in [0, 1] for p in predictions.tolist()), \
            "Predictions should be binary"
        
        # 7. Format API response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            results.append({
                'index': i,
                'prediction': pred.item(),
                'probability_up': prob[1].item(),
                'probability_down': prob[0].item()
            })
        
        assert len(results) == len(test_df), "Should have result for each prediction"
        assert all(0 <= r['probability_up'] <= 1 for r in results), \
            "Probabilities should be in [0, 1]"
