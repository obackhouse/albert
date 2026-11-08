"""Options."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


@dataclass(frozen=True)
class CostModel:
    """Cost model weights."""

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


@dataclass(frozen=True)
class ScalingLimits:
    """Scaling exponent limits."""

    time: dict[tuple[str, ...] | str, int] = field(default_factory=dict)
    memory: dict[tuple[str, ...] | str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate space keys and nonnegative exponent caps."""
        for name in ("time", "memory"):
            limits = getattr(self, name)
            for key, limit in limits.items():
                if isinstance(key, str):
                    valid_key = bool(key)
                else:
                    valid_key = (
                        isinstance(key, tuple) and bool(key) and all(isinstance(space, str) and space for space in key)
                    )
                if not valid_key:
                    raise ValueError(f"{name} keys must be a space name or nonempty tuple of names")
                if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                    raise ValueError(f"{name} limits must be nonnegative integers")


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
