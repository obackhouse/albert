"""Hashing utilities."""

from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING, TypeVar, Generic

if TYPE_CHECKING:
    from typing import Hashable

T = TypeVar("T")


class InternTable(Generic[T]):
    """A thread-safe table for interning objects."""

    def __init__(self):
        """Initialise the intern table."""
        self._table: weakref.WeakValueDictionary[Hashable, T] = weakref.WeakValueDictionary()
        self._lock = threading.RLock()

    def get(self, key: Hashable, factory: callable[[], T]) -> T:
        """Get an object from the table, creating it if necessary.

        Args:
            key: The key to look up.
            factory: A callable that creates the object if it is not found.

        Returns:
            The object associated with the key.
        """
        with self._lock:
            obj = self._table.get(key)
            if obj is None:
                obj = factory()
                self._table[key] = obj
            return obj

_INTERN_TABLE = InternTable[T]()
