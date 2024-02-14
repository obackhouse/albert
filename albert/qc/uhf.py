"""Expressions for unrestricted bases.
"""

from albert.qc.rhf import _make_symmetry
from albert.symmetry import Symmetry, antisymmetric_permutations
from albert.tensor import Symbol, Tensor


class UHFTensor(Tensor):
    """Tensor subclass for unrestricted bases."""

    def as_uhf(self):
        """Return an unrestricted representation of the object."""
        return self

    def as_rhf(self):
        """Return a restricted representation of the object."""
        pass

    def hashable(self):
        """Return a hashable representation of the object."""

        # Get the hashable representation
        hashable = super().hashable()

        # Add a penalty to prefer alternating spins
        spins = tuple(spin for index, spin in self.indices)
        penalty = sum(2 for i, spin in enumerate(spins) if spins[i - 1] == spin)
        if spins[0] != min(spins):
            penalty += 1
        hashable = (penalty,) + hashable[1:]

        return hashable


class UHFSymbol(Symbol):
    """Symbol subclass for unrestricted bases."""

    Tensor = UHFTensor

    def __getitem__(self, indices):
        """Return a tensor."""
        tensor = super().__getitem__(indices)
        tensor._symbol = self
        return tensor


class Hamiltonian1e(UHFSymbol):
    """Constructor for one-electron Hamiltonian-like symbols."""

    DESIRED_RANK = 2

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry(
            (0, 1),
            (1, 0),
        )


Fock = Hamiltonian1e("f")


class Hamiltonian2e(UHFSymbol):
    """Constructor for two-electron Hamiltonian-like symbols."""

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
        )


ERI = Hamiltonian2e("v")


class FermionicAmplitude(UHFSymbol):
    """Constructor for amplitude symbols."""

    def __init__(self, name, num_covariant, num_contravariant):
        """Initialise the object."""
        self.name = name
        self.DESIRED_RANK = num_covariant + num_contravariant
        perms = []
        for perm_covariant in antisymmetric_permutations(num_covariant):
            for perm_contravariant in antisymmetric_permutations(num_contravariant):
                perms.append(perm_covariant + perm_contravariant)
        self.symmetry = Symmetry(*perms)


T1 = FermionicAmplitude("t1", 1, 1)
T2 = FermionicAmplitude("t2", 2, 2)
T3 = FermionicAmplitude("t3", 3, 3)
