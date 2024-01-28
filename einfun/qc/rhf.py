"""Expressions for restricted Hartree--Fock bases.
"""

from einfun.symmetry import Permutation, Symmetry
from einfun.tensor import Symbol, Tensor


def _make_symmetry(*perms):
    """Make a symmetry from a list of permutations.
    """
    return Symmetry(*[Permutation(perm, 1) for perm in perms])


class Hamiltonian1e(Symbol):
    """Constructor for one-electron Hamiltonian-like symbols.
    """

    DESIRED_RANK = 2

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

    DESIRED_RANK = 4

    def __init__(self, name):
        """Initialise the object.
        """
        self.name = name
        # FIXME this is for real orbitals only
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


class FermionicAmplitude(Symbol):
    """Constructor for amplitude symbols.
    """

    def __init__(self, name, num_covariant, num_contravariant):
        """Initialise the object.
        """
        self.name = name
        self.DESIRED_RANK = num_covariant + num_contravariant
        # FIXME how to generalise?
        if (num_covariant, num_contravariant) == (2, 2):
            self.symmetry = _make_symmetry(
                (0, 1, 2, 3),
                (1, 0, 3, 2),
            )
        elif (num_covariant, num_contravariant) == (3, 3):
            self.symmetry = _make_symmetry(
                (0, 1, 2, 3, 4, 5),
                (2, 1, 0, 5, 4, 3),
            )
        else:
            self.symmetry = _make_symmetry(tuple(range(num_covariant + num_contravariant)))


T1 = FermionicAmplitude("t1", 1, 1)
T2 = FermionicAmplitude("t2", 2, 2)
T3 = FermionicAmplitude("t3", 3, 3)
