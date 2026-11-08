"""Options."""

from __future__ import annotations

from enum import IntEnum, StrEnum


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
