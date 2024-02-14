"""Expressions for restricted bases.
"""

from albert.symmetry import Permutation, Symmetry
from albert.tensor import Symbol, Tensor


_as_rhf = {}


class RHFTensor(Tensor):
    """Tensor subclass for restricted bases."""

    def as_uhf(self):
        """Return an unrestricted representation of the object."""
        raise NotImplementedError

    def as_rhf(self):
        """Return a restricted representation of the object."""
        symbol = self.as_symbol()
        if symbol not in _as_rhf:
            raise NotImplementedError(
                f"Conversion of `{symbol.__class__.__name__}` from unrestricted to "
                "restricted is not implemented."
            )
        return _as_rhf[symbol](self)


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


class Hamiltonian1e(Symbol):
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


def _Fock_as_rhf(tensor):
    """
    Convert a `Fock`-derived tensor object from generalised to
    unrestricted.
    """
    indices = tensor.indices
    assert all(spin in ("α", "β") for index, spin in indices)
    indices = tuple(index for index, spin in indices)
    return Fock[indices]


_as_rhf[Fock] = _Fock_as_rhf


class Hamiltonian2e(Symbol):
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
            (2, 3, 0, 1),
            (3, 2, 0, 1),
            (2, 3, 1, 0),
            (3, 2, 1, 0),
        )


ERI = Hamiltonian2e("v")


def _ERI_as_rhf(tensor):
    """
    Convert an `ERI`-derived tensor object from generalised to
    unrestricted.
    """
    indices = tensor.indices
    assert all(spin in ("α", "β") for index, spin in indices)
    indices = tuple(index for index, spin in indices)
    return ERI[indices]


class FermionicAmplitude(Symbol):
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
