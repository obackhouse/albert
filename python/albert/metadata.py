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

    spaces: dict[str, list[str]] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    symmetries: dict[str, list[Permutation]] = field(default_factory=dict)
    cost_factor: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for symbol, factor in self.cost_factor.items():
            if not symbol or isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
                raise ValueError("cost_factor values must be positive integers keyed by tensor symbol")

    def add_space(
        self,
        space: str,
        indices: Sequence[str],
        *,
        size: int = 8,
    ) -> None:
        """Add a space to the metadata."""
        self.spaces[space] = list(indices)
        self.sizes[space] = size

    def add_tensor(
        self,
        symbol: str,
        *,
        symmetry: Sequence[Permutation] | None = None,
        cost_factor: int = 1,
    ) -> None:
        """Add a tensor to the metadata."""
        self.cost_factor[symbol] = cost_factor
        if symmetry is not None:
            self.symmetries[symbol] = list(symmetry)
