"""Options."""

from __future__ import annotations

from enum import StrEnum


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
