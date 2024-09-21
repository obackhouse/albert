"""Expressions for generalised bases.
"""

import itertools

from albert.qc import uhf
from albert.qc.rhf import _make_symmetry
from albert.symmetry import Permutation, Symmetry, antisymmetric_permutations
from albert.tensor import Symbol, Tensor


class GHFTensor(Tensor):
    """Tensor subclass for generalised bases."""

    def as_uhf(self, target_restricted=False):
        """Return an unrestricted representation of the object."""
        return self._symbol._as_uhf(self, target_restricted=target_restricted)

    def as_rhf(self):
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


class ScalarSymbol(GHFSymbol):
    """Constructor for scalar symbols."""

    DESIRED_RANK = 0

    def __init__(self, name, uhf_symbol=None):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry()
        self.uhf_symbol = uhf_symbol

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """Convert a scalar tensor object from generalised to unrestricted."""
        return (tensor._symbol.uhf_symbol[tuple()],)


R0 = ScalarSymbol("r0", uhf.R0)
L0 = ScalarSymbol("l0", uhf.L0)


class FockSymbol(GHFSymbol):
    """Constructor for Fock-like symbols."""

    DESIRED_RANK = 2
    uhf_symbol = uhf.Fock

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry(
            (0, 1),
            (1, 0),
        )

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """
        Convert a `Fock`-derived tensor object from generalised to
        unrestricted.
        """

        # Loop over spins
        tensors = []
        for spin in ("α", "β"):
            # Check if first index has fixed spin
            if tensor.indices[0].spin and tensor.indices[0].spin != spin:
                continue

            # Check if second index has fixed spin
            if tensor.indices[1].spin and tensor.indices[1].spin != spin:
                continue

            # Get the UHF tensor part
            indices = tuple(index.to_spin(spin) for index in tensor.indices)
            tensors.append(tensor._symbol.uhf_symbol[indices])

        return tuple(tensors)


Fock = FockSymbol("f")


class BosonicHamiltonianSymbol(GHFSymbol):
    """Constructor for bosonic Hamiltonian symbols."""

    DESIRED_RANK = 1
    uhf_symbol = uhf.BosonicHamiltonian

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry((0,))

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """
        Convert a `BosonicHamiltonian`-derived tensor object from
        generalised to unrestricted.
        """
        return (tensor._symbol.uhf_symbol[tensor.indices],)


BosonicHamiltonian = BosonicHamiltonianSymbol("G")


class BosonicInteractionHamiltonianSymbol(GHFSymbol):
    """Constructor for bosonic interaction Hamiltonian symbols."""

    DESIRED_RANK = 2
    uhf_symbol = uhf.BosonicInteractionHamiltonian

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry((0, 1), (1, 0))

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """
        Convert a `BosonicInteractionHamiltonian`-derived tensor object
        from generalised to unrestricted.
        """
        return (tensor._symbol.uhf_symbol[tensor.indices],)


BosonicInteractionHamiltonian = BosonicInteractionHamiltonianSymbol("w")


class ElectronBosonHamiltonianSymbol(GHFSymbol):
    """Constructor for electron-boson Hamiltonian symbols."""

    DESIRED_RANK = 3

    def __init__(self, name, uhf_symbol):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry(
            (0, 1, 2),
            (0, 2, 1),
        )
        self.uhf_symbol = uhf_symbol

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """
        Convert a `ElectronBosonHamiltonian`-derived tensor object from
        generalised to unrestricted.
        """

        # Loop over spins
        tensors = []
        for spin in ("α", "β"):
            # Check if first index has fixed spin
            if tensor.indices[1].spin and tensor.indices[1].spin != spin:
                continue

            # Check if second index has fixed spin
            if tensor.indices[2].spin and tensor.indices[2].spin != spin:
                continue

            # Get the UHF tensor part
            indices = tuple(index.to_spin(spin) for index in tensor.indices[1:])
            indices = (tensor.indices[0],) + indices
            tensors.append(tensor._symbol.uhf_symbol[indices])

        return tuple(tensors)


ElectronBosonHamiltonian = ElectronBosonHamiltonianSymbol("g", uhf.ElectronBosonHamiltonian)
ElectronBosonConjHamiltonian = ElectronBosonHamiltonianSymbol(
    "gc", uhf.ElectronBosonConjHamiltonian
)


class RDM1Symbol(FockSymbol):
    """Constructor for one-electron reduced density matrix symbols."""

    uhf_symbol = uhf.RDM1


RDM1 = RDM1Symbol("d")


class DeltaSymbol(FockSymbol):
    """Constructor for the Kronecker delta symbol."""

    uhf_symbol = uhf.Delta


Delta = DeltaSymbol("δ")


class ERISymbol(GHFSymbol):
    """
    Constructor for antisymmetric two-electron integral symbols.
    """

    DESIRED_RANK = 4
    uhf_symbol = uhf.ERI

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

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """
        Convert a `ERI`-derived tensor object from generalised to
        unrestricted.

        Note: The result is in the chemist's notation.
        """

        # Loop over spins
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
            # Check if indices have fixed spins
            if any(index.spin and index.spin != spin for index, spin in zip(indices_bare, spins)):
                continue

            # Get the indices
            indices = tuple(index.to_spin(spin) for index, spin in zip(indices_bare, spins))

            # Get the UHF symbol
            uhf_symbol = tensor._symbol.uhf_symbol

            if direct:
                # Get the direct contribution
                uhf_tensor.append(uhf_symbol[indices[0], indices[2], indices[1], indices[3]])

            if exchange:
                # Get the exchange contribution
                uhf_tensor.append(-uhf_symbol[indices[0], indices[3], indices[1], indices[2]])

        return tuple(uhf_tensor)


ERI = ERISymbol("v")


class SingleERISymbol(GHFSymbol):
    """
    Constructor for non-antisymmetric two-electron integral symbols.
    """

    DESIRED_RANK = 4
    uhf_symbol = uhf.ERI

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        # FIXME this is for real orbitals only
        self.symmetry = Symmetry(
            (Permutation((0, 1, 2, 3), +1)),
            (Permutation((1, 0, 3, 2), +1)),
            (Permutation((2, 3, 0, 1), +1)),
            (Permutation((3, 2, 1, 0), +1)),
        )

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """
        Convert a `ERI`-derived tensor object from generalised to
        unrestricted.

        Note: The result is in the chemist's notation.
        """

        # Loop over spins
        uhf_tensor = []
        indices_bare = tensor.indices
        for spins in [
            ("αααα"),
            ("ββββ"),
            ("αβαβ"),
            ("βαβα"),
        ]:
            # Check if indices have fixed spins
            if any(index.spin and index.spin != spin for index, spin in zip(indices_bare, spins)):
                continue

            # Get the indices
            indices = tuple(index.to_spin(spin) for index, spin in zip(indices_bare, spins))

            # Get the UHF symbol
            uhf_symbol = tensor._symbol.uhf_symbol

            # Get the tensor
            uhf_tensor.append(uhf_symbol[indices[0], indices[2], indices[1], indices[3]])

        return tuple(uhf_tensor)


SingleERI = SingleERISymbol("vs")


class RDM2Symbol(GHFSymbol):
    """Constructor for two-electron reduced density matrix symbols."""

    DESIRED_RANK = 4
    uhf_symbol = uhf.RDM2

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = Symmetry(
            (Permutation((0, 1, 2, 3), +1)),
            (Permutation((0, 1, 3, 2), -1)),
            (Permutation((1, 0, 2, 3), -1)),
            (Permutation((1, 0, 3, 2), +1)),
        )

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """
        Convert a `RDM2`-derived tensor object from generalised to
        unrestricted.
        """

        # Loop over spins
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
            # Check if indices have fixed spins
            if any(index.spin and index.spin != spin for index, spin in zip(indices_bare, spins)):
                continue

            # Get the indices
            indices = tuple(index.to_spin(spin) for index, spin in zip(indices_bare, spins))

            # Get the UHF symbol
            uhf_symbol = tensor._symbol.uhf_symbol

            if direct:
                # Get the direct contribution
                uhf_tensor.append(uhf_symbol[indices[0], indices[1], indices[2], indices[3]])

            if exchange:
                # Get the exchange contribution
                uhf_tensor.append(-uhf_symbol[indices[0], indices[1], indices[3], indices[2]])

        return tuple(uhf_tensor)


RDM2 = RDM2Symbol("Γ")


class FermionicAmplitude(GHFSymbol):
    """Constructor for fermionic amplitude symbols."""

    uhf_symbol = None

    def __init__(self, name, num_covariant, num_contravariant, uhf_symbol=None):
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
        self.uhf_symbol = uhf_symbol

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """
        Convert a `Tn`-derived tensor object from generalised to
        unrestricted.
        """

        n = (tensor._symbol._num_covariant, tensor._symbol._num_contravariant)

        # Loop over spins
        uhf_tensor = []
        for covariant in itertools.product("αβ", repeat=max(n)):
            for contravariant in set(itertools.permutations(covariant)):
                if n[0] > n[1]:
                    spins = covariant[:n[0]] + contravariant[:n[1]]
                else:
                    spins = contravariant[:n[1]] + covariant[:n[0]]

                # Check if indices have fixed spins
                if any(
                    index.spin and index.spin != spin
                    for index, spin in zip(tensor.indices, spins)
                ):
                    continue

                # Get the UHF tensor part
                indices = tuple(index.to_spin(spin) for index, spin in zip(tensor.indices, spins))
                uhf_tensor_part = tensor._symbol.uhf_symbol[indices]

                if not target_restricted:
                    # Expand antisymmetry where spin allows
                    for perm in antisymmetric_permutations(max(n)):
                        if n[0] > n[1]:
                            full_perm = perm + Permutation(tuple(range(n[1])), 1)
                        else:
                            full_perm = Permutation(tuple(range(n[0])), 1) + perm
                        spins_perm = tuple(spins[i] for i in full_perm.permutation)
                        if spins == spins_perm:
                            uhf_tensor.append(uhf_tensor_part.permute_indices(full_perm))
                else:
                    uhf_tensor.append(uhf_tensor_part)

        return tuple(uhf_tensor)


T1 = FermionicAmplitude("t1", 1, 1, uhf_symbol=uhf.T1)
T2 = FermionicAmplitude("t2", 2, 2, uhf_symbol=uhf.T2)
T3 = FermionicAmplitude("t3", 3, 3, uhf_symbol=uhf.T3)
L1 = FermionicAmplitude("l1", 1, 1, uhf_symbol=uhf.L1)
L2 = FermionicAmplitude("l2", 2, 2, uhf_symbol=uhf.L2)
L3 = FermionicAmplitude("l3", 3, 3, uhf_symbol=uhf.L3)

R1ip = FermionicAmplitude("r1", 1, 0, uhf_symbol=uhf.R1ip)
R2ip = FermionicAmplitude("r2", 2, 1, uhf_symbol=uhf.R2ip)
R3ip = FermionicAmplitude("r3", 3, 2, uhf_symbol=uhf.R3ip)
R1ea = FermionicAmplitude("r1", 0, 1, uhf_symbol=uhf.R1ea)
R2ea = FermionicAmplitude("r2", 1, 2, uhf_symbol=uhf.R2ea)
R3ea = FermionicAmplitude("r3", 2, 3, uhf_symbol=uhf.R3ea)
R1ee = FermionicAmplitude("r1", 1, 1, uhf_symbol=uhf.R1ee)
R2ee = FermionicAmplitude("r2", 2, 2, uhf_symbol=uhf.R2ee)
R3ee = FermionicAmplitude("r3", 3, 3, uhf_symbol=uhf.R3ee)


class BosonicAmplitude(GHFSymbol):
    """Constructor for bosonic amplitude symbols."""

    uhf_symbol = None

    def __init__(self, name, num_bosons, uhf_symbol=None):
        """Initialise the object."""
        self.name = name
        self.DESIRED_RANK = num_bosons
        perms = []
        for perm in antisymmetric_permutations(num_bosons):
            perms.append(Permutation(perm.permutation, 1))
        self.symmetry = Symmetry(*perms)
        self.uhf_symbol = uhf_symbol

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """
        Convert a `Sn`-derived tensor object from generalised to
        unrestricted.
        """
        return (tensor._symbol.uhf_symbol[tensor.indices],)


S1 = BosonicAmplitude("s1", 1, uhf_symbol=uhf.S1)
S2 = BosonicAmplitude("s2", 2, uhf_symbol=uhf.S2)
LS1 = BosonicAmplitude("ls1", 1, uhf_symbol=uhf.LS1)
LS2 = BosonicAmplitude("ls2", 2, uhf_symbol=uhf.LS2)


class MixedAmplitude(GHFSymbol):
    """Constructor for mixed amplitude symbols."""

    uhf_symbol = None

    def __init__(self, name, num_bosons, num_covariant, num_contravariant, uhf_symbol=None):
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
        self.uhf_symbol = uhf_symbol

    @staticmethod
    def _as_uhf(tensor, target_restricted=False):
        """
        Convert a `Unm`-derived tensor object from generalised to
        unrestricted.
        """

        # FIXME this is just for U/LU amplitudes
        nb = tensor._symbol.NUM_BOSONS
        nf = (tensor.rank - tensor._symbol.NUM_BOSONS) // 2

        # Loop over spins
        uhf_tensor = []
        for covariant in itertools.product("αβ", repeat=nf):
            for contravariant in set(itertools.permutations(covariant)):
                # Check if indices have fixed spins
                if any(
                    index.spin and index.spin != spin
                    for index, spin in zip(tensor.indices[nb:], covariant + contravariant)
                ):
                    continue

                # Get the UHF tensor part
                spins = tuple(covariant) + tuple(contravariant)
                indices = tuple(
                    index.to_spin(spin) for index, spin in zip(tensor.indices[nb:], spins)
                )
                indices = tensor.indices[:nb] + indices
                uhf_tensor_part = tensor._symbol.uhf_symbol[indices]

                if not target_restricted:
                    # Expand antisymmetry where spin allows
                    for perm in antisymmetric_permutations(nf):
                        full_perm = Permutation(tuple(range(nb + nf)), 1) + perm
                        spins_perm = tuple(spins[i] for i in full_perm.permutation)
                        if spins == spins_perm:
                            uhf_tensor.append(uhf_tensor_part.permute_indices(full_perm))
                else:
                    uhf_tensor.append(uhf_tensor_part)

        return tuple(uhf_tensor)


U11 = MixedAmplitude("u11", 1, 1, 1, uhf_symbol=uhf.U11)
U12 = MixedAmplitude("u12", 2, 1, 1, uhf_symbol=uhf.U12)
LU11 = MixedAmplitude("lu11", 1, 1, 1, uhf_symbol=uhf.LU11)
LU12 = MixedAmplitude("lu12", 2, 1, 1, uhf_symbol=uhf.LU12)
