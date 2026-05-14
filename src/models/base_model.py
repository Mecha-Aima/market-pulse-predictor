from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseSequentialModel(nn.Module, ABC):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.head = nn.Linear(hidden_size, output_size)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass."""

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

