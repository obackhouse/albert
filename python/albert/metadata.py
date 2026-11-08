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
