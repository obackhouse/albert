"""Options."""

from __future__ import annotations

from enum import StrEnum


class Traversal(StrEnum):
    """Traversal options."""

    PREORDER = "preorder"
    POSTORDER = "postorder"
