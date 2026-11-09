"""Tensor metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from albert._core import Permutation


@dataclass(frozen=True)
class Metadata:
    """Tensor metadata."""

    spaces: dict[str, list[str]] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    symmetries: dict[str, list[Permutation]] = field(default_factory=dict)
    cost_factor: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate iterative cost factors."""
        for symbol, factor in self.cost_factor.items():
            if not symbol or isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
                raise ValueError("cost_factor values must be positive integers keyed by tensor symbol")
