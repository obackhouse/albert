"""Tensor metadata."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Metadata:
    """Tensor metadata."""

    spaces: dict[str, list[str]] = field(default_factory=dict)
