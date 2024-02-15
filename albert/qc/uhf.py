"""Expressions for unrestricted bases.
"""

from albert.qc import rhf
from albert.qc.rhf import _make_symmetry
from albert.symmetry import Symmetry, antisymmetric_permutations
from albert.tensor import Symbol, Tensor

_as_rhf = {}


class UHFTensor(Tensor):
    """Tensor subclass for unrestricted bases."""

    def as_uhf(self, *args, **kwargs):
        """Return an unrestricted representation of the object."""
        return self

    def as_rhf(self, *args, **kwargs):
        """Return a restricted representation of the object."""
        symbol = self.as_symbol()
        if symbol not in _as_rhf:
            raise NotImplementedError(
                f"Conversion of `{symbol.__class__.__name__}` from unrestricted to "
                "restricted is not implemented."
            )
        return _as_rhf[symbol](self)

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


def _Fock_as_rhf(tensor):
    """
    Convert a `Fock`-derived tensor object from generalised to
    unrestricted.
    """
    indices = tensor.indices
    assert all(spin in ("α", "β") for index, spin in indices)
    indices = tuple(index for index, spin in indices)
    return rhf.Fock[indices]


_as_rhf[Fock] = _Fock_as_rhf


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


def _ERI_as_rhf(tensor):
    """
    Convert an `ERI`-derived tensor object from generalised to
    unrestricted.
    """
    indices = tensor.indices
    assert all(spin in ("α", "β") for index, spin in indices)
    indices = tuple(index for index, spin in indices)
    return rhf.ERI[indices]


_as_rhf[ERI] = _ERI_as_rhf


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


def _gen_Tn_as_rhf(n, Tn_rhf, Tn_uhf):
    """
    Generate a function to convert a `Tn`-derived tensor object from
    unrestricted to restricted.
    """

    def _Tn_as_rhf(tensor):
        """
        Convert a `Tn`-derived tensor object from unrestricted to
        restricted.
        """

        # Check input
        assert all(spin in ("α", "β") for index, spin in tensor.indices)

        # Spin flip if needed
        nα = sum(spin == "α" for index, spin in tensor.indices)
        nβ = sum(spin == "β" for index, spin in tensor.indices)
        if nβ > nα:
            indices = tuple((index, {"α": "β", "β": "α"}[spin]) for index, spin in tensor.indices)
            tensor = tensor.copy(*indices)

        # Expand same spin contributions as linear combinations of
        # mixed spin contributions
        tensors = [tensor]
        if tensor.rank > 2:
            if all(spin == "α" for index, spin in tensor.indices):
                # Get the spins of the tensors in the linear combination
                spins = []
                for k in range(n):
                    spin = [("α", "β")[j % 2] for j in range(n)]
                    spin += [("α", "β")[j == k] for j in range(n)]
                    spins.append(spin)

                # Get the new tensors
                tensors = []
                for spin in spins:
                    indices = tuple(
                        (index, new_spin) for (index, spin), new_spin in zip(tensor.indices, spin)
                    )
                    tensors.append(Tn_uhf[indices])

        # Relabel the indices
        tensor = 0
        for t in tensors:
            indices = tuple(index for index, spin in t.external_indices)
            tensor += Tn_rhf[indices]

        return tensor

    return _Tn_as_rhf


_as_rhf[T1] = _gen_Tn_as_rhf(1, rhf.T1, T1)
_as_rhf[T2] = _gen_Tn_as_rhf(2, rhf.T2, T2)
_as_rhf[T3] = _gen_Tn_as_rhf(3, rhf.T3, T3)
