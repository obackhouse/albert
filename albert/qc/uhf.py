"""Expressions for unrestricted bases.
"""

from albert.algebra import Mul
from albert.base import Base
from albert.qc import rhf
from albert.qc.rhf import _make_symmetry
from albert.symmetry import Symmetry, antisymmetric_permutations
from albert.tensor import Symbol, Tensor


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

    def as_json(self):
        """Return a JSON serialisable representation of the object."""
        return {
            "_type": self.__class__.__name__,
            "_path": self.__module__,
            "index": self.index,
            "spin": self.spin,
        }

    @classmethod
    def from_json(cls, data):
        """Return an object from a JSON serialisable representation.

        Notes
        -----
        This method is non-recursive and the dictionary members should
        already be parsed.
        """
        return cls(data["index"], data["spin"])

    def _prepare_other(self, other):
        """Prepare the other object."""
        if not isinstance(other, SpinIndex):
            return SpinIndex(other, "")
        return other

    def __repr__(self):
        """Return a string representation of the object."""
        return f"{self.index}{self.spin}"

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
        return self._symbol._as_rhf(self)

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


class FockSymbol(UHFSymbol):
    """Constructor for one-electron Fock-like symbols."""

    DESIRED_RANK = 2
    rhf_symbol = rhf.Fock

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry(
            (0, 1),
            (1, 0),
        )

    @staticmethod
    def _as_rhf(tensor):
        """
        Convert a `Fock`-derived tensor object from generalised to
        unrestricted.
        """
        assert all(isinstance(index, SpinIndex) for index in tensor.indices)
        indices = tuple(index.index for index in tensor.indices)
        return tensor._symbol.rhf_symbol[indices]


Fock = FockSymbol("f")


class RDM1Symbol(FockSymbol):
    """Constructor for one-electron reduced density matrix symbols."""

    rhf_symbol = rhf.RDM1


RDM1 = RDM1Symbol("d")


class DeltaSymbol(FockSymbol):
    """Constructor for the Kronecker delta symbol."""

    rhf_symbol = rhf.Delta


Delta = DeltaSymbol("δ")


class ERISymbol(UHFSymbol):
    """Constructor for two-electron integral-like symbols."""

    DESIRED_RANK = 4
    rhf_symbol = rhf.ERI

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

    @staticmethod
    def _as_rhf(tensor):
        """
        Convert an `ERI`-derived tensor object from generalised to
        unrestricted.
        """
        assert all(isinstance(index, SpinIndex) for index in tensor.indices)
        indices = tuple(index.index for index in tensor.indices)
        return tensor._symbol.rhf_symbol[indices]


ERI = ERISymbol("v")


class CDERISymbol(UHFSymbol):
    """
    Constructor for Cholesky-decomposed two-electron integral-like
    symbols.
    """

    DESIRED_RANK = 3
    rhf_symbol = rhf.CDERI

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        # FIXME this is for real orbitals only
        self.symmetry = _make_symmetry(
            (0, 1, 2),
            (0, 2, 1),
        )

    @staticmethod
    def _as_rhf(tensor):
        """
        Convert an `ERI`-derived tensor object from generalised to
        unrestricted.
        """
        assert all(isinstance(index, SpinIndex) for index in tensor.indices[1:])
        indices = (tensor.indices[0],) + tuple(index.index for index in tensor.indices[1:])
        return tensor._symbol.rhf_symbol[indices]


CDERI = CDERISymbol("v")


class RDM2Symbol(ERISymbol):
    """Constructor for two-electron reduced density matrix symbols."""

    rhf_symbol = rhf.RDM2

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry((0, 1, 2, 3))


RDM2 = RDM2Symbol("Γ")


class FermionicAmplitude(UHFSymbol):
    """Constructor for amplitude symbols."""

    rhf_symbol = None

    def __init__(self, name, num_covariant, num_contravariant, rhf_symbol=None):
        """Initialise the object."""
        self.name = name
        self.DESIRED_RANK = num_covariant + num_contravariant
        perms = []
        for perm_covariant in antisymmetric_permutations(num_covariant):
            for perm_contravariant in antisymmetric_permutations(num_contravariant):
                perms.append(perm_covariant + perm_contravariant)
        self.symmetry = Symmetry(*perms)
        self.rhf_symbol = rhf_symbol

    @staticmethod
    def _as_rhf(tensor):
        """
        Convert a `Tn`-derived tensor object from unrestricted to
        restricted.
        """

        # FIXME this is just for T/L amplitudes
        n = tensor.rank // 2

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
                    tensors.append(tensor._symbol[indices])

        # Canonicalise the indices
        tensors = [t.canonicalise().expand() for t in tensors]

        # Relabel the indices
        tensor_out = 0
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
            tensor_out += tensor._symbol.rhf_symbol[indices] * factor

        return tensor_out.expand()


T1 = FermionicAmplitude("t1", 1, 1, rhf_symbol=rhf.T1)
T2 = FermionicAmplitude("t2", 2, 2, rhf_symbol=rhf.T2)
T3 = FermionicAmplitude("t3", 3, 3, rhf_symbol=rhf.T3)
L1 = FermionicAmplitude("l1", 1, 1, rhf_symbol=rhf.L1)
L2 = FermionicAmplitude("l2", 2, 2, rhf_symbol=rhf.L2)
L3 = FermionicAmplitude("l3", 3, 3, rhf_symbol=rhf.L3)
