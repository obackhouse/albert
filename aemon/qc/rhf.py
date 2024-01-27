"""Expressions for restricted Hartree--Fock bases.
"""

from aemon.symmetry import Permutation, Symmetry
from aemon.tensor import Symbol, Tensor


def _make_symmetry(*perms):
    """Make a symmetry from a list of permutations.
    """
    return Symmetry(*[Permutation(perm, 1) for perm in perms])


class Hamiltonian1e(Symbol):
    """Constructor for one-electron Hamiltonian-like symbols.
    """

    def __init__(self, name):
        """Initialise the object.
        """
        self.name = name
        self.symmetry = _make_symmetry(
            (0, 1),
            (1, 0),
        )


Fock = Hamiltonian1e("f")


class Hamiltonian2e(Symbol):
    """Constructor for two-electron Hamiltonian-like symbols.
    """

    def __init__(self, name):
        """Initialise the object.
        """
        self.name = name
        self.symmetry = _make_symmetry(
            (0, 1, 2, 3),
            (0, 1, 3, 2),
            (1, 0, 2, 3),
            (1, 0, 3, 2),
            (2, 3, 0, 1),
            (3, 2, 0, 1),
            (2, 3, 1, 0),
            (3, 2, 1, 0),
        )


ERI = Hamiltonian2e("v")
