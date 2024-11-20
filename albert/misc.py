"""Miscellaneous utility functions."""

from __future__ import annotations

import re
import time
from collections.abc import Container
from typing import TYPE_CHECKING, Hashable

if TYPE_CHECKING:
    from typing import Any, Iterable, Optional

    from albert.base import Base


class ExclusionSet(Container[Hashable]):
    """A set that is defined by its exclusions rather than its inclusions."""

    def __init__(self, exclusions: Optional[Iterable[Hashable]] = None) -> None:
        """Initialise the object."""
        self._exclusions = set(exclusions) if exclusions is not None else set()

    def __contains__(self, item: Any) -> bool:
        """Check if an item is in the set.

        Args:
            item: The item to check.

        Returns:
            Whether the item is in the set.
        """
        return item not in self._exclusions

    def add(self, item: Hashable) -> None:
        """Add an item to the set.

        Args:
            item: The item to add.

        Note:
            This method removes the item from the exclusions.
        """
        self._exclusions.discard(item)

    def discard(self, item: Hashable) -> None:
        """Remove an item from the set.

        Args:
            item: The item to remove.

        Note:
            This method adds the item to the exclusions.
        """
        self._exclusions.add(item)

    def remove(self, item: Hashable) -> None:
        """Remove an item from the set.

        Args:
            item: The item to remove.

        Raises:
            KeyError: If the item is not in the set.
        """
        if item not in self:
            raise KeyError(item)
        self.discard(item)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}({self._exclusions})"


class Stopwatch:
    """A simple stopwatch for timing code execution."""

    def __init__(self, name: Optional[str] = None):
        """Initialise the object."""
        self._name = name
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self) -> Stopwatch:
        """Start the stopwatch."""
        self._start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the stopwatch."""
        self._end = time.time()
        if self._name is not None:
            print(f"{self._name}: {self.elapsed:.3f} s")

    @property
    def elapsed(self) -> float:
        """Return the elapsed time in seconds."""
        if self._start is None:
            raise RuntimeError("Stopwatch has not been started")
        start = self._start
        if self._end is None:
            end = time.time()
        else:
            end = self._end
        return end - start


def from_string(string: str) -> Base:
    """Convert an object from a string representation to the algebraic object.

    Args:
        string: The string representation of the object.

    Returns:
        The algebraic object.

    Notes:
        This function is intended for convenience when debugging and testing. It lacks many
        features offered by directly instantiating the object. One pitfall in particular is that
        tensor names cannot contain numbers.
    """

    # Find all distinct indices
    tensors = [(name, inds.split(",")) for name, inds in re.findall(r"(\w+)\(([^)]+)\)", string)]
    indices = set(index for _, inds in tensors for index in inds)

    def _format_scalar(m: re.Match[str]) -> str:
        """Format a scalar statement from a matched regular expression."""
        value = m.group(1)
        return f"Scalar({value})"

    def _format_tensor(m: re.Match[str]) -> str:
        """Format a tensor statement from a matched regular expression."""
        name = m.group(1)
        inds = m.group(2).split(",")
        inds_string = ", ".join(f'"{x.strip()}"' for x in inds)
        return f'Tensor(*from_list([{inds_string}]), name="{name}")'

    # Make the substitutions
    string = re.sub(r"(\w+)\(([^)]+)\)", _format_tensor, string)
    string = re.sub(r"(\d+\.\d+|\d+)", _format_scalar, string)

    # Evaluate the string
    expr: Base = eval(string)

    return expr
