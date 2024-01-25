"""Permutation symmetry.
"""

from aemon.base import Base


class Permutation(Base):
    """Permutation.
    """

    def __init__(self, permutation, sign):
        """Initialise the object.
        """
        self.permutation = permutation
        self.sign = sign

    def __call__(self, tensor):
        """Apply the permutation to the tensor.
        """
        indices = tensor.external_indices
        permuted_indices = [indices[i] for i in self.permutation]
        return tensor.map_indices(dict(zip(indices, permuted_indices)))

    def hashable(self):
        """Return a hashable representation of the object.
        """
        return (self.permutation, self.sign)


class Symmetry(Base):
    """Permutation symmetry.
    """

    def __init__(self, *permutations):
        """Initialise the object.
        """
        self.permutations = permutations

    def __call__(self, tensor):
        """Iterate over the permutations of the tensor.
        """
        for permutation in self.permutations:
            yield permutation(tensor)

    def hashable(self):
        """Return a hashable representation of the object.
        """
        return tuple(permutation.hashable() for permutation in self.permutations)
