"""
Integration tests for model training pipeline.

Tests model training with real data (1 epoch for speed).
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.rnn_model import SimpleRNNModel
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel


@pytest.mark.integration
class TestModelTrainingIntegration:
    """Integration tests for model training with real data."""
    
    def test_train_rnn_one_epoch(self, fixture_features_data, tmp_path):
        """Test: Model training (1 epoch) with RNN."""
        # Prepare data
        features_df = fixture_features_data.copy()
        
        # Create simple train/test split
        split_idx = int(len(features_df) * 0.8)
        train_df = features_df[:split_idx]
        test_df = features_df[split_idx:]
        
        if len(train_df) < 10 or len(test_df) < 5:
            pytest.skip("Not enough data for training")
        
        # Prepare tensors (simplified)
        feature_cols = ['close', 'volume', 'returns']
        available_cols = [col for col in feature_cols if col in train_df.columns]
        
        if len(available_cols) == 0:
            pytest.skip("No numeric features available")
        
        X_train = torch.tensor(train_df[available_cols].values, dtype=torch.float32)
        y_train = torch.tensor(train_df['label'].values if 'label' in train_df.columns else np.random.randint(0, 2, len(train_df)), dtype=torch.long)
        
        # Reshape for RNN (batch, seq_len, features)
        X_train = X_train.unsqueeze(1)  # Add sequence dimension
        
        # Initialize model
        input_size = len(available_cols)
        model = SimpleRNNModel(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            num_classes=2
        )
        
        # Simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        
        # Verify training
        assert isinstance(train_loss, float), "Should return loss"
        assert train_loss > 0, "Loss should be positive"
        assert not np.isnan(train_loss), "Loss should not be NaN"
    
    def test_train_lstm_one_epoch(self, fixture_features_data):
        """Test: Model training (1 epoch) with LSTM."""
        features_df = fixture_features_data.copy()
        
        split_idx = int(len(features_df) * 0.8)
        train_df = features_df[:split_idx]
        
        if len(train_df) < 10:
            pytest.skip("Not enough data for training")
        
        # Prepare data
        feature_cols = ['close', 'volume']
        available_cols = [col for col in feature_cols if col in train_df.columns]
        
        if len(available_cols) == 0:
            pytest.skip("No numeric features available")
        
        X_train = torch.tensor(train_df[available_cols].values, dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(train_df['label'].values if 'label' in train_df.columns else np.random.randint(0, 2, len(train_df)), dtype=torch.long)
        
        # Initialize LSTM model
        model = LSTMModel(
            input_size=len(available_cols),
            hidden_size=32,
            num_layers=1,
            num_classes=2
        )
        
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        
        # Verify
        assert isinstance(train_loss, float), "Should return loss"
        assert not np.isnan(train_loss), "Loss should not be NaN"
    
    def test_train_gru_one_epoch(self, fixture_features_data):
        """Test: Model training (1 epoch) with GRU."""
        features_df = fixture_features_data.copy()
        
        split_idx = int(len(features_df) * 0.8)
        train_df = features_df[:split_idx]
        
        if len(train_df) < 10:
            pytest.skip("Not enough data for training")
        
        # Prepare data
        feature_cols = ['close', 'volume']
        available_cols = [col for col in feature_cols if col in train_df.columns]
        
        if len(available_cols) == 0:
            pytest.skip("No numeric features available")
        
        X_train = torch.tensor(train_df[available_cols].values, dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(train_df['label'].values if 'label' in train_df.columns else np.random.randint(0, 2, len(train_df)), dtype=torch.long)
        
        # Initialize GRU model
        model = GRUModel(
            input_size=len(available_cols),
            hidden_size=32,
            num_layers=1,
            num_classes=2
        )
        
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        
        # Verify
        assert isinstance(train_loss, float), "Should return loss"
        assert not np.isnan(train_loss), "Loss should not be NaN"


@pytest.mark.integration
class TestModelEvaluationIntegration:
    """Integration tests for model evaluation."""
    
    def test_evaluate_trained_model(self, fixture_features_data):
        """Test: Model evaluation after training."""
        features_df = fixture_features_data.copy()
        
        split_idx = int(len(features_df) * 0.8)
        train_df = features_df[:split_idx]
        test_df = features_df[split_idx:]
        
        if len(train_df) < 10 or len(test_df) < 5:
            pytest.skip("Not enough data for evaluation")
        
        # Prepare data
        feature_cols = ['close', 'volume']
        available_cols = [col for col in feature_cols if col in train_df.columns]
        
        if len(available_cols) == 0:
            pytest.skip("No numeric features available")
        
        X_train = torch.tensor(train_df[available_cols].values, dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(train_df['label'].values if 'label' in train_df.columns else np.random.randint(0, 2, len(train_df)), dtype=torch.long)
        
        X_test = torch.tensor(test_df[available_cols].values, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(test_df['label'].values if 'label' in test_df.columns else np.random.randint(0, 2, len(test_df)), dtype=torch.long)
        
        # Train model
        model = SimpleRNNModel(
            input_size=len(available_cols),
            hidden_size=32,
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
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            predictions = torch.argmax(test_outputs, dim=1)
            accuracy = (predictions == y_test).float().mean()
        
        # Verify metrics
        assert test_loss.item() >= 0, "Loss should be non-negative"
        assert 0 <= accuracy.item() <= 1, "Accuracy should be in [0, 1]"
    
    def test_model_prediction(self, fixture_features_data):
        """Test: Model prediction on new data."""
        features_df = fixture_features_data.copy()
        
        if len(features_df) < 10:
            pytest.skip("Not enough data")
        
        # Prepare data
        feature_cols = ['close', 'volume']
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        if len(available_cols) == 0:
            pytest.skip("No numeric features available")
        
        X = torch.tensor(features_df[available_cols].values, dtype=torch.float32).unsqueeze(1)
        
        # Create and train model
        model = SimpleRNNModel(
            input_size=len(available_cols),
            hidden_size=32,
            num_layers=1,
            num_classes=2
        )
        
        # Make predictions (even without training)
        model.eval()
        with torch.no_grad():
            predictions = model(X)
        
        # Verify predictions
        assert predictions.shape[0] == len(X), "Should predict for all samples"
        assert predictions.shape[1] == 2, "Should have 2 class probabilities"
        
        # Verify probabilities sum to ~1 (after softmax)
        probs = torch.softmax(predictions, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(len(X)), atol=1e-5), \
            "Probabilities should sum to 1"


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for complete training pipeline."""
    
    def test_end_to_end_training_pipeline(self, fixture_features_data, tmp_path):
        """Test: Complete training pipeline from features to saved model."""
        features_df = fixture_features_data.copy()
        
        split_idx = int(len(features_df) * 0.8)
        train_df = features_df[:split_idx]
        test_df = features_df[split_idx:]
        
        if len(train_df) < 10 or len(test_df) < 5:
            pytest.skip("Not enough data")
        
        # 1. Prepare data
        feature_cols = ['close', 'volume']
        available_cols = [col for col in feature_cols if col in train_df.columns]
        
        if len(available_cols) == 0:
            pytest.skip("No numeric features available")
        
        X_train = torch.tensor(train_df[available_cols].values, dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(train_df['label'].values if 'label' in train_df.columns else np.random.randint(0, 2, len(train_df)), dtype=torch.long)
        
        X_test = torch.tensor(test_df[available_cols].values, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(test_df['label'].values if 'label' in test_df.columns else np.random.randint(0, 2, len(test_df)), dtype=torch.long)
        
        # 2. Initialize model
        model = SimpleRNNModel(
            input_size=len(available_cols),
            hidden_size=32,
            num_layers=1,
            num_classes=2
        )
        
        # 3. Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        
        # 4. Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            predictions = torch.argmax(test_outputs, dim=1)
            accuracy = (predictions == y_test).float().mean()
        
        # 5. Save model
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # 6. Verify saved model
        assert model_path.exists(), "Model file should be created"
        
        # 7. Load and verify
        loaded_model = SimpleRNNModel(
            input_size=len(available_cols),
            hidden_size=32,
            num_layers=1,
            num_classes=2
        )
        loaded_model.load_state_dict(torch.load(model_path))
        
        # 8. Verify loaded model works
        loaded_model.eval()
        with torch.no_grad():
            predictions = loaded_model(X_test[:5])
        
        assert predictions.shape[0] == 5, "Should predict for 5 samples"
        
        # Verify metrics
        assert train_loss > 0, "Should have training loss"
        assert 0 <= accuracy.item() <= 1, "Should have valid accuracy"
