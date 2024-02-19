"""Base class.
"""


class Base:
    """Base class."""

    def __hash__(self):
        """Return the hash of the object."""
        return hash(self.hashable())

    def _prepare_other(self, other):
        """Prepare the other object."""
        if not isinstance(other, Base):
            raise TypeError(
                f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}."
            )
        return other

    def __eq__(self, other):
        """Compare two objects."""
        other = self._prepare_other(other)
        if not isinstance(other, Base):
            return False
        return self.hashable() == other.hashable()

    def __ne__(self, other):
        """Compare two objects."""
        other = self._prepare_other(other)
        if not isinstance(other, Base):
            return True
        return self.hashable() != other.hashable()

    def __lt__(self, other):
        """Compare two objects."""
        other = self._prepare_other(other)
        return self.hashable() < other.hashable()

    def __le__(self, other):
        """Compare two objects."""
        other = self._prepare_other(other)
        return self.hashable() <= other.hashable()

    def __gt__(self, other):
        """Compare two objects."""
        other = self._prepare_other(other)
        return self.hashable() > other.hashable()

    def __ge__(self, other):
        """Compare two objects."""
        other = self._prepare_other(other)
        return self.hashable() >= other.hashable()
