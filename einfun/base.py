"""Base class.
"""


class Base:
    """Base class."""

    def __hash__(self):
        """Return the hash of the object."""
        return hash(self.hashable())

    def __eq__(self, other):
        """Compare two objects."""
        if not isinstance(other, Base):
            return False
        return self.hashable() == other.hashable()

    def __ne__(self, other):
        """Compare two objects."""
        if not isinstance(other, Base):
            return True
        return self.hashable() != other.hashable()

    def __lt__(self, other):
        """Compare two objects."""
        return self.hashable() < other.hashable()

    def __le__(self, other):
        """Compare two objects."""
        return self.hashable() <= other.hashable()

    def __gt__(self, other):
        """Compare two objects."""
        return self.hashable() > other.hashable()

    def __ge__(self, other):
        """Compare two objects."""
        return self.hashable() >= other.hashable()
