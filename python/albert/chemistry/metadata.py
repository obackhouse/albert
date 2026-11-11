"""Tensor metadata for chemistry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from albert.chemistry.options import SUPPORTED_TENSOR_SYMBOLS, OrbitalType
from albert.metadata import Metadata as BaseMetadata

if TYPE_CHECKING:
    from typing import Self

    from albert._core import Permutation


def get_default_symmetries() -> dict[str, list[Permutation]]:
    """Get the default symmetries for chemistry tensors."""
    return {
        tensor.symbol: tensor.symmetries
        for tensor in SUPPORTED_TENSOR_SYMBOLS.values()
        if tensor.symmetries is not None
    }


def get_default_indices() -> dict[str, list[str]]:
    """Get the default indices for chemistry tensors."""
    return {
        "o": ["i", "j", "k", "l", "o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7"],
        "v": ["a", "b", "c", "d", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
        "x": ["P", "Q", "R", "S", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"],
    }


def get_default_sizes() -> dict[str, int]:
    """Get the default sizes for chemistry tensors."""
    return {
        "o": 64,
        "v": 512,
        "x": 1728,
    }


@dataclass
class Metadata(BaseMetadata):
    """Tensor metadata for chemistry."""

    indices: dict[str, list[str]] = field(default_factory=get_default_indices)
    """Indices of the spaces."""

    sizes: dict[str, int] = field(default_factory=get_default_sizes)
    """Sizes of the spaces."""

    symmetries: dict[str, list[Permutation]] = field(default_factory=get_default_symmetries)
    """Tensor symmetries keyed by tensor symbol."""

    def with_spin(self) -> Self:
        """Return a copy of the metadata with spin spaces added."""
        orbital_types = [OrbitalType.ALPHA, OrbitalType.BETA, OrbitalType.SPIN]
        for space, indices in self.indices.items():
            if any(index.endswith(orbital.subscript) for index in indices for orbital in orbital_types):
                raise ValueError(f"space {space} already has spin-aware indices")
        metadata = type(self)(
            indices=self.indices.copy(),
            sizes=self.sizes.copy(),
            symmetries=self.symmetries.copy(),
            cost_factors=self.cost_factors.copy(),
        )

        # Update the spaces
        for space, indices in self.indices.items():
            size = self.sizes[space]
            for orbital in orbital_types:
                metadata.add_space(
                    space + orbital.subscript,
                    [index + orbital.subscript for index in indices],
                    size=size * orbital.size_factor,
                )

        # Verify the metadata
        metadata.verify()

        return metadata
