"""Tensor metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from albert._core import Permutation


@dataclass
class Metadata:
    """Tensor metadata."""

    indices: dict[str, list[str]] = field(default_factory=dict)
    """Indices of the spaces."""

    sizes: dict[str, int] = field(default_factory=dict)
    """Sizes of the spaces."""

    symmetries: dict[str, list[Permutation]] = field(default_factory=dict)
    """Tensor symmetries keyed by tensor symbol."""

    cost_factors: dict[str, int] = field(default_factory=dict)
    """Tensor cost factors keyed by tensor symbol."""

    def __post_init__(self) -> None:
        self.verify()

    def verify(self) -> None:
        """Verify the metadata."""
        if self.indices.keys() != self.sizes.keys():
            raise ValueError("indices and sizes must have the same keys")
        all_indices: set[str] = set()
        for space, indices in self.indices.items():
            indices_set = set(indices)
            if all_indices & indices_set:
                raise ValueError(f"indices for space {space} must not overlap with other spaces")
            if len(indices_set) != len(indices):
                raise ValueError(f"indices for space {space} must be unique")
            all_indices.update(indices_set)
        for symbol in self.symmetries.keys() | self.cost_factors.keys():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("tensor metadata keys must be non-empty tensor symbols")
        for factor in self.cost_factors.values():
            if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
                raise ValueError("cost_factors values must be positive integers")

    def add_space(
        self,
        space: str,
        indices: Sequence[str],
        *,
        size: int = 8,
    ) -> None:
        """Add a space to the metadata."""
        self.indices[space] = list(indices)
        self.sizes[space] = size
        self.verify()

    def add_tensor(
        self,
        symbol: str,
        *,
        symmetry: Sequence[Permutation] | None = None,
        cost_factor: int | None = None,
    ) -> None:
        """Add a tensor to the metadata."""
        if symmetry is not None:
            self.symmetries[symbol] = list(symmetry)
        if cost_factor is not None:
            self.cost_factors[symbol] = cost_factor
        self.verify()
