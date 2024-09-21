"""Expressions for unrestricted bases.
"""

from albert.algebra import Mul
from albert.qc import rhf
from albert.qc.rhf import _make_symmetry
from albert.symmetry import Permutation, Symmetry, antisymmetric_permutations
from albert.tensor import Symbol, Tensor


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
            spins = tuple(getattr(index, "spin", "") for index in tensor.indices)
            penalty = 0
            for i in range(len(spins) - 1):
                penalty += int(spins[i] == spins[i + 1]) * 2
            if spins[0] != min(spins):
                penalty += 1
            return penalty

        # Get the hashable representation
        hashable = Tensor.hashable(self, penalty_function=_penalty)

        return hashable

    def canonicalise(self):
        """Canonicalise the object."""
        if not self.symmetry:
            return self
        candidates = list(self.symmetry(self))
        # FIXME this is super hacky
        if hasattr(self._symbol, "_num_covariant"):
            new_candidates = []
            for i in range(len(candidates)):
                spins_covariant = tuple(index.spin for index in candidates[i].external_indices[: self._symbol._num_covariant])
                spins_contravariant = tuple(index.spin for index in candidates[i].external_indices[self._symbol._num_covariant :])
                n = min(len(spins_covariant), len(spins_contravariant))
                if spins_covariant[:n] == spins_contravariant[:n]:
                    new_candidates.append(candidates[i])
            candidates = new_candidates
        return min(candidates)


class UHFSymbol(Symbol):
    """Symbol subclass for unrestricted bases."""

    Tensor = UHFTensor

    def __getitem__(self, indices):
        """Return a tensor."""
        tensor = super().__getitem__(indices)
        tensor._symbol = self
        return tensor


class ScalarSymbol(UHFSymbol):
    """Constructor for scalar symbols."""

    DESIRED_RANK = 0

    def __init__(self, name, rhf_symbol=None):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry(())
        self.rhf_symbol = rhf_symbol

    @staticmethod
    def _as_rhf(tensor):
        """
        Convert a `Scalar`-derived tensor object from unrestricted to
        restricted.
        """
        return tensor._symbol.rhf_symbol[tuple()]


R0 = ScalarSymbol("r0", rhf.R0)
L0 = ScalarSymbol("l0", rhf.L0)


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
        assert all(index.spin for index in tensor.indices)
        indices = tuple(index.to_spin(None) for index in tensor.indices)
        return tensor._symbol.rhf_symbol[indices]


Fock = FockSymbol("f")


class BosonicHamiltonianSymbol(UHFSymbol):
    """Constructor for bosonic Hamiltonian symbols."""

    DESIRED_RANK = 1
    rhf_symbol = rhf.BosonicHamiltonian

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry((0,))

    @staticmethod
    def _as_rhf(tensor):
        """
        Convert a `BosonicHamiltonian`-derived tensor object from
        unrestricted to restricted.
        """
        return tensor


BosonicHamiltonian = BosonicHamiltonianSymbol("G")


class BosonicInteractionHamiltonianSymbol(UHFSymbol):
    """Constructor for bosonic interaction Hamiltonian symbols."""

    DESIRED_RANK = 2
    rhf_symbol = rhf.BosonicInteractionHamiltonian

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry((0, 1), (1, 0))

    @staticmethod
    def _as_rhf(tensor):
        """
        Convert a `BosonicInteractionHamiltonian`-derived tensor object
        from unrestricted to restricted.
        """
        return tensor


BosonicInteractionHamiltonian = BosonicInteractionHamiltonianSymbol("w")


class ElectronBosonHamiltonianSymbol(UHFSymbol):
    """Constructor for electron-boson Hamiltonian symbols."""

    DESIRED_RANK = 3

    def __init__(self, name, rhf_symbol):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry(
            (0, 1, 2),
            (0, 2, 1),
        )
        self.rhf_symbol = rhf_symbol

    @staticmethod
    def _as_rhf(tensor):
        """
        Convert a `ElectronBosonHamiltonian`-derived tensor object from
        unrestricted to restricted.
        """
        assert all(index.spin for index in tensor.indices[1:])
        indices = tuple(index.to_spin(None) for index in tensor.indices[1:])
        indices = (tensor.indices[0],) + indices
        return tensor._symbol.rhf_symbol[indices]


ElectronBosonHamiltonian = ElectronBosonHamiltonianSymbol("g", rhf.ElectronBosonHamiltonian)
ElectronBosonConjHamiltonian = ElectronBosonHamiltonianSymbol(
    "gc", rhf.ElectronBosonConjHamiltonian
)


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
        assert all(index.spin for index in tensor.indices)
        indices = tuple(index.to_spin(None) for index in tensor.indices)
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
        assert all(index.spin for index in tensor.indices[1:])
        indices = (tensor.indices[0],) + tuple(index.to_spin(None) for index in tensor.indices[1:])
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
        self._num_covariant = num_covariant
        self._num_contravariant = num_contravariant
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

        n = (tensor._symbol._num_covariant, tensor._symbol._num_contravariant)

        # Check input
        assert all(index.spin for index in tensor.indices)

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
                for k in range(n[0]):
                    spin = [("α", "β")[j % 2] for j in range(n[0])]
                    spin += [("α", "β")[j == k] for j in range(n[1])]
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
            indices = tuple(index.to_spin(None) for index in t.indices)
            tensor_out += tensor._symbol.rhf_symbol[indices] * factor

        return tensor_out.expand()


T1 = FermionicAmplitude("t1", 1, 1, rhf_symbol=rhf.T1)
T2 = FermionicAmplitude("t2", 2, 2, rhf_symbol=rhf.T2)
T3 = FermionicAmplitude("t3", 3, 3, rhf_symbol=rhf.T3)
L1 = FermionicAmplitude("l1", 1, 1, rhf_symbol=rhf.L1)
L2 = FermionicAmplitude("l2", 2, 2, rhf_symbol=rhf.L2)
L3 = FermionicAmplitude("l3", 3, 3, rhf_symbol=rhf.L3)

R1ip = FermionicAmplitude("r1", 1, 0, rhf_symbol=rhf.R1ip)
R2ip = FermionicAmplitude("r2", 2, 1, rhf_symbol=rhf.R2ip)
R3ip = FermionicAmplitude("r3", 3, 2, rhf_symbol=rhf.R3ip)
R1ea = FermionicAmplitude("r1", 0, 1, rhf_symbol=rhf.R1ea)
R2ea = FermionicAmplitude("r2", 1, 2, rhf_symbol=rhf.R2ea)
R3ea = FermionicAmplitude("r3", 2, 3, rhf_symbol=rhf.R3ea)
R1ee = FermionicAmplitude("r1", 1, 1, rhf_symbol=rhf.R1ee)
R2ee = FermionicAmplitude("r2", 2, 2, rhf_symbol=rhf.R2ee)
R3ee = FermionicAmplitude("r3", 3, 3, rhf_symbol=rhf.R3ee)


class BosonicAmplitude(UHFSymbol):
    """Constructor for bosonic amplitude symbols."""

    rhf_symbol = None

    def __init__(self, name, num_bosons, rhf_symbol=None):
        """Initialise the object."""
        self.name = name
        self.DESIRED_RANK = num_bosons
        perms = []
        for perm in antisymmetric_permutations(num_bosons):
            perms.append(Permutation(perm.permutation, 1))
        self.symmetry = Symmetry(*perms)
        self.rhf_symbol = rhf_symbol

    @staticmethod
    def _as_rhf(tensor):
        """
        Convert a `Sn`-derived tensor object from unrestricted to
        restricted.
        """
        return tensor


S1 = BosonicAmplitude("s1", 1, rhf_symbol=rhf.S1)
S2 = BosonicAmplitude("s2", 2, rhf_symbol=rhf.S2)
LS1 = BosonicAmplitude("ls1", 1, rhf_symbol=rhf.LS1)
LS2 = BosonicAmplitude("ls2", 2, rhf_symbol=rhf.LS2)


class MixedAmplitude(UHFSymbol):
    """Constructor for mixed amplitude symbols."""

    rhf_symbol = None

    def __init__(self, name, num_bosons, num_covariant, num_contravariant, rhf_symbol=None):
        """Initialise the object."""
        self.name = name
        self.DESIRED_RANK = num_bosons + num_covariant + num_contravariant
        self.NUM_BOSONS = num_bosons
        perms = []
        for perm_boson in antisymmetric_permutations(num_bosons):
            perm_boson = Permutation(perm_boson.permutation, 1)
            for perm_covariant in antisymmetric_permutations(num_covariant):
                for perm_contravariant in antisymmetric_permutations(num_contravariant):
                    perms.append(perm_boson + perm_covariant + perm_contravariant)
        self.symmetry = Symmetry(*perms)
        self.rhf_symbol = rhf_symbol

    @staticmethod
    def _as_rhf(tensor):
        """
        Convert a `Tn`-derived tensor object from unrestricted to
        restricted.
        """

        # FIXME this is just for T/L amplitudes
        nb = tensor._symbol.NUM_BOSONS
        nf = (tensor.rank - tensor._symbol.NUM_BOSONS) // 2

        # Check input
        assert all(index.spin for index in tensor.indices[nb:])

        # Spin flip if needed
        nα = sum(index.spin == "α" for index in tensor.indices[nb:])
        nβ = sum(index.spin == "β" for index in tensor.indices[nb:])
        if nβ > nα:
            indices = tuple(index.spin_flip() for index in tensor.indices[nb:])
            indices = tensor.indices[:nb] + indices
            tensor = tensor.copy(*indices)

        # Expand same spin contributions as linear combinations of
        # mixed spin contributions
        tensors = [tensor]
        if tensor.rank > 2:
            if all(index.spin == "α" for index in tensor.indices[nb:]):
                # Get the spins of the tensors in the linear combination
                spins = []
                for k in range(nf):
                    spin = [("α", "β")[j % 2] for j in range(nf)]
                    spin += [("α", "β")[j == k] for j in range(nf)]
                    spins.append(spin)

                # Get the new tensors
                tensors = []
                for spin in spins:
                    indices = tuple(index.to_spin(s) for index, s in zip(tensor.indices[nb:], spin))
                    indices = tensor.indices[:nb] + indices
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
            indices = tuple(index.to_spin(None) for index in t.indices[nb:])
            indices = tensor.indices[:nb] + indices
            tensor_out += tensor._symbol.rhf_symbol[indices] * factor

        return tensor_out.expand()


U11 = MixedAmplitude("u11", 1, 1, 1, rhf_symbol=rhf.U11)
U12 = MixedAmplitude("u12", 2, 1, 1, rhf_symbol=rhf.U12)
LU11 = MixedAmplitude("lu11", 1, 1, 1, rhf_symbol=rhf.LU11)
LU12 = MixedAmplitude("lu12", 2, 1, 1, rhf_symbol=rhf.LU12)
