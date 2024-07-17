"""Indices for quantum chemistry.
"""

from albert.base import Base


class Index(Base):
    """Index class."""

    def __init__(self, name, spin=None, space=None):
        """Initialise the object."""
        self.name = name
        self._spin = spin
        self.space = space

    def hashable(self):
        """Return a hashable representation of the object."""
        return (self.space if self.space else "", self.spin if self.spin else "", self.name)

    def as_json(self):
        """Return a JSON serialisable representation of the object."""
        return {
            "_type": self.__class__.__name__,
            "_path": self.__module__,
            "name": self.name,
            "spin": self.spin,
            "space": self.space,
        }

    @classmethod
    def from_json(cls, data):
        """Return an object from a JSON serialisable representation.

        Notes
        -----
        This method is non-recursive and the dictionary members should
        already be parsed.
        """
        return cls(data["name"], data["spin"], data["space"])

    def _prepare_other(self, other):
        """Prepare the other object."""
        if not isinstance(other, Index):
            return Index(other)
        return other

    def __repr__(self):
        """Return a string representation of the object."""
        return f"{self.name}{self.spin if self.spin else ''}"

    def to_spin(self, spin):
        """Return a copy of the object with the spin set to `spin`."""
        return Index(self.name, spin, self.space)

    def spin_flip(self):
        """Return a copy of the object with the spin flipped."""
        return self.to_spin({"α": "β", "β": "α", "": None}[self.spin])

    def to_space(self, space):
        """Return a copy of the object with the space set to `space`."""
        return Index(self.name, self.spin, space)

    @property
    def spin(self):
        """Return the spin of the object."""
        if not self._spin:
            return ""
        return self._spin

    @spin.setter
    def spin(self, value):
        """Set the spin of the object."""
        self._spin = value
