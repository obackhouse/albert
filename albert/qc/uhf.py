"""Expressions for unrestricted bases.
"""

from albert.algebra import Mul
from albert.base import Base
from albert.qc import rhf
from albert.qc.rhf import _make_symmetry
from albert.symmetry import Symmetry, antisymmetric_permutations
from albert.tensor import Symbol, Tensor

_as_rhf = {}


class SpinIndex(Base):
    """
    Class to represent a spin index. Adds a `spin` attribute to any
    accepted index type.
    """

    def __init__(self, index, spin):
        """Initialise the object."""
        self.index = index
        self.spin = spin

    def hashable(self):
        """Return a hashable representation of the object."""
        return (self.spin, self.index)

    def __repr__(self):
        """Return a string representation of the object."""
        return f"{self.index}({self.spin})"

    def to_spin(self, spin):
        """Return a copy of the object with the spin set to `spin`."""
        return SpinIndex(self.index, spin)

    def spin_flip(self):
        """Return a copy of the object with the spin flipped."""
        return self.to_spin({"α": "β", "β": "α"}[self.spin])


class UHFTensor(Tensor):
    """Tensor subclass for unrestricted bases."""

    def as_uhf(self):
        """Return an unrestricted representation of the object."""
        return self

    def as_rhf(self):
        """Return a restricted representation of the object."""
        symbol = self.as_symbol()
        if symbol not in _as_rhf:
            raise NotImplementedError(
                f"Conversion of `{symbol.__class__.__name__}` from unrestricted to "
                "restricted is not implemented."
            )
        return _as_rhf[symbol](self)

    def hashable(self, penalty_function=None):
        """Return a hashable representation of the object."""

        # Get the penalty function
        def _penalty(tensor):
            spins = tuple(index.spin for index in tensor.indices)
            penalty = sum(2 for i, spin in enumerate(spins) if spins[i - 1] == spin)
            if spins[0] != min(spins):
                penalty += 1
            return penalty

        # Get the hashable representation
        hashable = Tensor.hashable(self, penalty_function=_penalty)

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
    assert all(isinstance(index, SpinIndex) for index in tensor.indices)
    indices = tuple(index.index for index in tensor.indices)
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
    assert all(isinstance(index, SpinIndex) for index in tensor.indices)
    indices = tuple(index.index for index in tensor.indices)
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
        assert all(isinstance(index, SpinIndex) for index in tensor.indices)

        # Spin flip if needed
        nα = sum(index.spin == "α" for index in tensor.indices)
        nβ = sum(index.spin == "β" for index in tensor.indices)
        if nβ > nα:
            indices = tuple(index.spin_flip() for index in tensor.indices)
            tensor = tensor.copy(*indices)

        # Expand same spin contributions as linear combinations of
        # mixed spin contributions
        tensors = [tensor]
        if tensor.rank > 2:
            if all(index.spin == "α" for index in tensor.indices):
                # Get the spins of the tensors in the linear combination
                spins = []
                for k in range(n):
                    spin = [("α", "β")[j % 2] for j in range(n)]
                    spin += [("α", "β")[j == k] for j in range(n)]
                    spins.append(spin)

                # Get the new tensors
                tensors = []
                for spin in spins:
                    indices = tuple(index.to_spin(s) for index, s in zip(tensor.indices, spin))
                    tensors.append(Tn_uhf[indices])

        # Canonicalise the indices
        tensors = [t.canonicalise().expand() for t in tensors]

        # Relabel the indices
        tensor = 0
        for t in tensors:
            # The canonicalisation may have introduced a factor
            if isinstance(t, Mul):
                factor = t.coefficient
                args = t.without_coefficient().args
                assert len(args) == 1
                t = args[0]
            else:
                factor = 1

            # Get the restricted tensor
            indices = tuple(index.index for index in t.indices)
            tensor += Tn_rhf[indices] * factor

        return tensor.expand()

    return _Tn_as_rhf


_as_rhf[T1] = _gen_Tn_as_rhf(1, rhf.T1, T1)
_as_rhf[T2] = _gen_Tn_as_rhf(2, rhf.T2, T2)
_as_rhf[T3] = _gen_Tn_as_rhf(3, rhf.T3, T3)
