"""Expressions for restricted bases.
"""

from albert.symmetry import Permutation, Symmetry
from albert.tensor import Symbol, Tensor


class RHFTensor(Tensor):
    """Tensor subclass for restricted bases."""

    def as_uhf(self):
        """Return an unrestricted representation of the object."""
        raise NotImplementedError

    def as_rhf(self):
        """Return a restricted representation of the object."""
        raise NotImplementedError


class RHFSymbol(Symbol):
    """Symbol subclass for restricted bases."""

    Tensor = RHFTensor

    def __getitem__(self, indices):
        """Return a tensor."""
        tensor = super().__getitem__(indices)
        tensor._symbol = self
        return tensor


def _make_symmetry(*perms):
    """Make a symmetry from a list of permutations."""
    return Symmetry(*[Permutation(perm, 1) for perm in perms])


class FockSymbol(RHFSymbol):
    """Constructor for one-electron Fock-like symbols."""

    DESIRED_RANK = 2

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry(
            (0, 1),
            (1, 0),
        )


Fock = FockSymbol("f")


class RDM1Symbol(RHFSymbol):
    """Constructor for one-electron reduced density matrix symbols."""

    pass


RDM1 = RDM1Symbol("d")


class DeltaSymbol(RHFSymbol):
    """Constructor for the Kronecker delta symbol."""

    pass


Delta = DeltaSymbol("δ")


class ERISymbol(RHFSymbol):
    """Constructor for two-electron Fock-like symbols."""

    DESIRED_RANK = 4

    def __init__(self, name):
        """Initialise the object."""
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


ERI = ERISymbol("v")


class CDERISymbol(RHFSymbol):
    """
    Constructor for Cholesky-decomposed two-electron integral-like
    symbols.
    """

    DESIRED_RANK = 3

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        # FIXME this is for real orbitals only
        self.symmetry = _make_symmetry(
            (0, 1, 2),
            (0, 2, 1),
        )


CDERI = CDERISymbol("v")


class RDM2Symbol(ERISymbol):
    """Constructor for two-electron reduced density matrix symbols."""

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry((0, 1, 2, 3))


RDM2 = RDM2Symbol("Γ")


class FermionicAmplitude(RHFSymbol):
    """Constructor for amplitude symbols."""

    def __init__(self, name, num_covariant, num_contravariant):
        """Initialise the object."""
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
