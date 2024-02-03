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
        indices = tensor.external_indices
        permuted_indices = [indices[i] for i in self.permutation]
        return tensor.map_indices(dict(zip(indices, permuted_indices)))

    def hashable(self):
        """Return a hashable representation of the object."""
        return (self.permutation, self.sign)

    def __add__(self, other):
        """Append permutations."""
        perm = self.permutation + tuple(p + len(self.permutation) for p in other.permutation)
        sign = self.sign * other.sign
        return Permutation(perm, sign)


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
