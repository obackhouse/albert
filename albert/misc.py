"""Miscellaneous utility functions."""

from __future__ import annotations

from collections.abc import Container
from typing import TYPE_CHECKING, Hashable

if TYPE_CHECKING:
    from typing import Any, Iterable, Optional


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
