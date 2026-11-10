"""Factories."""

from __future__ import annotations

from dataclasses import dataclass

from albert._core import Tensor


@dataclass(frozen=True)
class TensorFactory:
    """Factory for creating tensors."""

    symbol: str

    def __getitem__(self, indices: str | tuple[str, ...]) -> Tensor:
        if isinstance(indices, str):
            indices = (indices,)
        return Tensor(self.symbol, indices)
