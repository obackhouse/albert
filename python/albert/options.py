"""Options."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum, StrEnum


@dataclass(frozen=True)
class CostModel:
    """Relative weights for arithmetic and memory traffic in a cost estimate."""

    multiply: float = 1.0
    add: float = 1.0
    read: float = 0.1
    write: float = 0.1

    def __post_init__(self) -> None:
        """Reject weights that cannot form a useful nonnegative cost."""
        for name, weight in (
            ("multiply", self.multiply),
            ("add", self.add),
            ("read", self.read),
            ("write", self.write),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} cost weight must be finite and nonnegative")


class PhaseFactor(IntEnum):
    """Tensor symmetry phase factors."""

    POSITIVE = 1
    NEGATIVE = -1


class NodeKind(StrEnum):
    """AST node kinds."""

    SCALAR = "scalar"
    TENSOR = "tensor"
    ADD = "add"
    MUL = "mul"


class SearchOrder(StrEnum):
    """Traversal options."""

    PREORDER = "preorder"
    POSTORDER = "postorder"
