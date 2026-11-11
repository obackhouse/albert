"""Factories for chemistry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import TYPE_CHECKING

from albert.chemistry.options import SUPPORTED_TENSOR_SYMBOLS
from albert.factory import TensorFactory as BaseTensorFactory

if TYPE_CHECKING:
    from albert._core import Permutation, Tensor


@dataclass(frozen=True)
class TensorFactory(BaseTensorFactory):
    """Factory for creating tensors for chemistry."""

    def __post_init__(self) -> None:
        if self.symbol not in SUPPORTED_TENSOR_SYMBOLS:
            raise ValueError(f"unsupported tensor symbol {self.symbol}")

    @cached_property
    def rank(self) -> int:
        """Rank of the tensor."""
        return SUPPORTED_TENSOR_SYMBOLS[self.symbol].rank

    @cached_property
    def description(self) -> str:
        """Description of the tensor."""
        return SUPPORTED_TENSOR_SYMBOLS[self.symbol].description

    @cache
    def symmetry(self) -> list[Permutation] | None:
        """Symmetry of the tensor."""
        return SUPPORTED_TENSOR_SYMBOLS[self.symbol].symmetries

    def __getitem__(self, indices: str | tuple[str, ...]) -> Tensor:
        if isinstance(indices, str):
            indices = (indices,)
        if len(indices) != self.rank:
            raise ValueError(f"tensor {self.symbol} expects {self.rank} indices, got {len(indices)}")
        return BaseTensorFactory.__getitem__(self, indices)
