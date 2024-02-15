"""Expressions for generalised bases.
"""

import itertools

from albert.qc import uhf
from albert.qc.rhf import _make_symmetry
from albert.symmetry import Permutation, Symmetry, antisymmetric_permutations
from albert.tensor import Symbol, Tensor

_as_uhf = {}


class GHFTensor(Tensor):
    """Tensor subclass for generalised bases."""

    def as_uhf(self, *args, **kwargs):
        """Return an unrestricted representation of the object."""
        symbol = self.as_symbol()
        if symbol not in _as_uhf:
            raise NotImplementedError(
                f"Conversion of `{symbol.__class__.__name__}` from generalised to "
                "unrestricted is not implemented."
            )
        return _as_uhf[symbol](self)

    def as_rhf(self, *args, **kwargs):
        """Return a restricted representation of the object."""
        raise NotImplementedError(
            "Direct conversion of generalised to restricted is not implemented."
        )


class GHFSymbol(Symbol):
    """Symbol subclass for generalised bases."""

    Tensor = GHFTensor

    def __getitem__(self, indices):
        """Return a tensor."""
        tensor = super().__getitem__(indices)
        tensor._symbol = self
        return tensor


class Hamiltonian1e(GHFSymbol):
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


def _Fock_as_uhf(tensor):
    """
    Convert a `Fock`-derived tensor object from generalised to
    unrestricted.
    """
    indices_α = (uhf.SpinIndex(tensor.indices[0], "α"), uhf.SpinIndex(tensor.indices[1], "α"))
    indices_β = (uhf.SpinIndex(tensor.indices[0], "β"), uhf.SpinIndex(tensor.indices[1], "β"))
    return (uhf.Fock[indices_α], uhf.Fock[indices_β])


_as_uhf[Fock] = _Fock_as_uhf


class Hamiltonian2e(GHFSymbol):
    """
    Constructor for antisymmetric two-electron Hamiltonian-like
    symbols.
    """

    DESIRED_RANK = 4

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        # FIXME this is for real orbitals only
        self.symmetry = Symmetry(
            (Permutation((0, 1, 2, 3), +1)),
            (Permutation((0, 1, 3, 2), -1)),
            (Permutation((1, 0, 2, 3), -1)),
            (Permutation((1, 0, 3, 2), +1)),
            (Permutation((2, 3, 0, 1), +1)),
            (Permutation((3, 2, 0, 1), -1)),
            (Permutation((2, 3, 1, 0), -1)),
            (Permutation((3, 2, 1, 0), +1)),
        )


ERI = Hamiltonian2e("v")


def _ERI_as_uhf(tensor):
    """
    Convert a `ERI`-derived tensor object from generalised to
    unrestricted.

    Note: The result is in the chemist's notation.
    """

    uhf_tensor = []
    indices_bare = tensor.indices
    for spins, direct, exchange in [
        ("αααα", True, True),
        ("ββββ", True, True),
        ("αβαβ", True, False),
        ("βαβα", True, False),
        ("αββα", False, True),
        ("βααβ", False, True),
    ]:
        indices = tuple(uhf.SpinIndex(index, spin) for index, spin in zip(indices_bare, spins))

        if direct:
            uhf_tensor.append(uhf.ERI[indices[0], indices[2], indices[1], indices[3]])

        if exchange:
            uhf_tensor.append(-uhf.ERI[indices[0], indices[3], indices[1], indices[2]])

    return tuple(uhf_tensor)


_as_uhf[ERI] = _ERI_as_uhf


class FermionicAmplitude(GHFSymbol):
    """Constructor for amplitude symbols."""

    def __init__(self, name, num_covariant, num_contravariant):
        """Initialise the object."""
        self.name = name
        self.DEISRED_RANK = num_covariant + num_contravariant
        perms = []
        for perm_covariant in antisymmetric_permutations(num_covariant):
            for perm_contravariant in antisymmetric_permutations(num_contravariant):
                perms.append(perm_covariant + perm_contravariant)
        self.symmetry = Symmetry(*perms)


T1 = FermionicAmplitude("t1", 1, 1)
T2 = FermionicAmplitude("t2", 2, 2)
T3 = FermionicAmplitude("t3", 3, 3)


def _gen_Tn_as_uhf(n, Tn_uhf):
    """
    Generate a function to convert a `Tn`-derived tensor object from
    generalised to unrestricted.
    """

    def _Tn_as_uhf(tensor):
        """
        Convert a `Tn`-derived tensor object from generalised to
        unrestricted.
        """

        uhf_tensor = []
        for covariant in itertools.product("αβ", repeat=n):
            for contravariant in set(itertools.permutations(covariant)):
                # Get the UHF tensor part
                spins = tuple(covariant) + tuple(contravariant)
                indices = tuple(
                    uhf.SpinIndex(index, spin) for index, spin in zip(tensor.indices, spins)
                )
                uhf_tensor_part = Tn_uhf[indices]

                # Expand antisymmetry where spin allows
                for perm in antisymmetric_permutations(n):
                    full_perm = Permutation(tuple(range(n)), 1) + perm
                    spins_perm = tuple(spins[i] for i in full_perm.permutation)
                    if spins == spins_perm:
                        uhf_tensor.append(uhf_tensor_part.permute_indices(full_perm))

        return tuple(uhf_tensor)

    return _Tn_as_uhf


_as_uhf[T1] = _gen_Tn_as_uhf(1, uhf.T1)
_as_uhf[T2] = _gen_Tn_as_uhf(2, uhf.T2)
_as_uhf[T3] = _gen_Tn_as_uhf(3, uhf.T3)
