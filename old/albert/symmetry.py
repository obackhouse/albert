"""Permutation symmetry.
"""

from albert.base import Base


class Permutation(Base):
    """Permutation."""

    def __init__(self, permutation, sign):
        """Initialise the object."""
        self.permutation = tuple(permutation)
        self.sign = sign

    def __call__(self, tensor):
        """Apply the permutation to the tensor."""
        return tensor.permute_indices(self)

    def hashable(self):
        """Return a hashable representation of the object."""
        return (self.permutation, self.sign)

    def as_json(self):
        """Return a JSON representation of the object."""
        return {
            "_type": self.__class__.__name__,
            "_path": self.__module__,
            "permutation": self.permutation,
            "sign": self.sign,
        }

    @classmethod
    def from_json(cls, data):
        """Return an object from a JSON serialisable representation.

        Notes
        -----
        This method is non-recursive and the dictionary members should
        already be parsed.
        """
        return cls(data["permutation"], data["sign"])

    def __add__(self, other):
        """Append permutations."""
        perm = self.permutation + tuple(p + len(self.permutation) for p in other.permutation)
        sign = self.sign * other.sign
        return Permutation(perm, sign)

    def __repr__(self):
        """Return a string representation of the object."""
        return f"Permutation({self.permutation}, {self.sign})"


class Symmetry(Base):
    """Permutation symmetry."""

    def __init__(self, *permutations):
        """Initialise the object."""
        self.permutations = permutations

    def __call__(self, tensor):
        """Iterate over the permutations of the tensor."""
        for permutation in self.permutations:
            yield permutation(tensor)

    def hashable(self):
        """Return a hashable representation of the object."""
        return tuple(permutation.hashable() for permutation in self.permutations)

    def as_json(self):
        """Return a JSON representation of the object."""
        return {
            "_type": self.__class__.__name__,
            "_path": self.__module__,
            "permutations": [permutation.as_json() for permutation in self.permutations],
        }

    @classmethod
    def from_json(cls, data):
        """Return an object from a JSON serialisable representation.

        Notes
        -----
        This method is non-recursive and the dictionary members should
        already be parsed.
        """
        return cls(*data["permutations"])


def antisymmetric_permutations(n):
    """
    Return permutations of `n` objects with a sign equal to +1 for an
    even number of swaps and -1 for an odd number of swaps.

    Parameters
    ----------
    n : int
        Number of objects.

    Returns
    -------
    perms : list of Permutation
        Permutations.
    """

    def _permutations(seq):
        if not seq:
            return [[]]

        items = []
        for i, item in enumerate(_permutations(seq[:-1])):
            inds = range(len(item) + 1)
            if i % 2 == 0:
                inds = reversed(inds)
            items += [item[:i] + seq[-1:] + item[i:] for i in inds]

        return items

    return [
        Permutation(item, -1 if i % 2 else 1)
        for i, item in enumerate(_permutations(list(range(n))))
    ]
