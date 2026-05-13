from __future__ import annotations

from torch import Tensor, nn

from src.models.base_model import BaseSequentialModel


class GRUModel(BaseSequentialModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: Tensor) -> Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])

