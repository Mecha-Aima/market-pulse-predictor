import pytest

# Skip all tests in this module - Phase 4 tests not ready for CI yet
pytestmark = pytest.mark.skip(reason="Phase 4 model tests - will be enabled when Phase 4 is complete")

import torch

try:
    from src.models.gru_model import GRUModel
    from src.models.lstm_model import LSTMModel
    from src.models.rnn_model import SimpleRNNModel
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


def test_rnn_forward_pass_shape() -> None:
    model = SimpleRNNModel(
        input_size=10,
        hidden_size=8,
        num_layers=2,
        output_size=3,
        dropout=0.1,
    )

    output = model(torch.randn(32, 24, 10))

    assert output.shape == (32, 3)


def test_lstm_forward_pass_shape() -> None:
    model = LSTMModel(
        input_size=10,
        hidden_size=8,
        num_layers=2,
        output_size=3,
        dropout=0.1,
    )

    output = model(torch.randn(32, 24, 10))

    assert output.shape == (32, 3)


def test_gru_forward_pass_shape() -> None:
    model = GRUModel(
        input_size=10,
        hidden_size=8,
        num_layers=2,
        output_size=3,
        dropout=0.1,
    )

    output = model(torch.randn(32, 24, 10))

    assert output.shape == (32, 3)
