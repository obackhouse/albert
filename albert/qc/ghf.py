"""Classes for GHF tensors."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from albert.qc import uhf
from albert.symmetry import Permutation, Symmetry, fully_antisymmetric_group, symmetric_group
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Optional

    from albert.base import Base
    from albert.index import Index


# Developer notes:
# * There is a lot of code repetition and subclasses could easily be used, but I felt like this
#   prevented optimisations and makes the code more complicated when new tensors don't conform
#   to the same pattern.


class Fock(Tensor):
    """Class for the Fock matrix.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 2:
            raise ValueError("Fock matrix must have two indices.")
        if name is None:
            name = "f"
        if symmetry is None:
            symmetry = symmetric_group((0, 1), (1, 0))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # Loop over spins
        tensors = []
        for spin in ("a", "b"):
            # Check for fixed spins
            if any(index.spin and index.spin != spin for index in self.indices):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index in self.indices)
            tensors.append(uhf.Fock(*indices, name=self.name))

        return tuple(tensors)


class RDM1(Fock):
    """Class for the one-particle reduced density matrix.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 2:
            raise ValueError("One-particle reduced density matrix must have two indices.")
        if name is None:
            name = "d"
        if symmetry is None:
            symmetry = symmetric_group((0, 1), (1, 0))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # Loop over spins
        tensors = []
        for spin in ("a", "b"):
            # Check for fixed spins
            if any(index.spin and index.spin != spin for index in self.indices):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index in self.indices)
            tensors.append(uhf.RDM1(*indices, name=self.name))

        return tuple(tensors)


class Delta(Fock):
    """Class for the Kronecker delta.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 2:
            raise ValueError("Kronecker delta must have two indices.")
        if name is None:
            name = "δ"
        if symmetry is None:
            symmetry = symmetric_group((0, 1), (1, 0))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # Loop over spins
        tensors = []
        for spin in ("a", "b"):
            # Check for fixed spins
            if any(index.spin and index.spin != spin for index in self.indices):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index in self.indices)
            tensors.append(uhf.Delta(*indices, name=self.name))

        return tuple(tensors)


class ERI(Tensor):
    """Class for the electron repulsion integral tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 4:
            raise ValueError("ERI tensor must have four indices.")
        if name is None:
            name = "v"
        if symmetry is None:
            symmetry = Symmetry(
                Permutation((0, 1, 2, 3), +1),
                Permutation((0, 1, 3, 2), -1),
                Permutation((1, 0, 2, 3), -1),
                Permutation((1, 0, 3, 2), +1),
                Permutation((2, 3, 0, 1), +1),
                Permutation((3, 2, 0, 1), -1),
                Permutation((2, 3, 1, 0), -1),
                Permutation((3, 2, 1, 0), +1),
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # Loop over spins
        tensors: list[Base] = []
        for spins, direct, exchange in [
            ("aaaa", True, True),
            ("bbbb", True, True),
            ("abab", True, False),
            ("baba", True, False),
            ("abba", False, True),
            ("baab", False, True),
        ]:
            # Check for fixed spins
            if any(index.spin and index.spin != spin for index, spin in zip(self.indices, spins)):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index, spin in zip(self.indices, spins))
            if direct:
                tensors.append(
                    uhf.ERI(indices[0], indices[2], indices[1], indices[3], name=self.name)
                )
            if exchange:
                tensors.append(
                    -uhf.ERI(indices[0], indices[3], indices[1], indices[2], name=self.name)
                )

        return tuple(tensors)


class ERISingle(Tensor):
    """Class for the non-antisymmetric electron repulsion integral tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 4:
            raise ValueError("ERI tensor must have four indices.")
        if name is None:
            name = "v"
        if symmetry is None:
            symmetry = Symmetry(
                Permutation((0, 1, 2, 3), +1),
                Permutation((1, 0, 3, 2), +1),
                Permutation((2, 3, 0, 1), +1),
                Permutation((3, 2, 1, 0), +1),
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # Loop over spins
        tensors = []
        for spins in ["aaaa", "bbbb", "abab", "baba"]:
            # Check for fixed spins
            if any(index.spin and index.spin != spin for index, spin in zip(self.indices, spins)):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index, spin in zip(self.indices, spins))
            tensors.append(uhf.ERI(indices[0], indices[2], indices[1], indices[3], name=self.name))

        return tuple(tensors)


class RDM2(Tensor):
    """Class for the two-particle reduced density matrix.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 4:
            raise ValueError("RDM2 tensor must have four indices.")
        if name is None:
            name = "Γ"
        if symmetry is None:
            symmetry = Symmetry(
                Permutation((0, 1, 2, 3), +1),
                Permutation((0, 1, 3, 2), -1),
                Permutation((1, 0, 2, 3), -1),
                Permutation((1, 0, 3, 2), +1),
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # Loop over spins
        tensors: list[Base] = []
        for spins, direct, exchange in [
            ("aaaa", True, True),
            ("bbbb", True, True),
            ("abab", True, False),
            ("baba", True, False),
            ("abba", False, True),
            ("baab", False, True),
        ]:
            # Check for fixed spins
            if any(index.spin and index.spin != spin for index, spin in zip(self.indices, spins)):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index, spin in zip(self.indices, spins))
            if direct:
                tensors.append(
                    uhf.RDM2(indices[0], indices[1], indices[2], indices[3], name=self.name)
                )
            if exchange:
                tensors.append(
                    -uhf.RDM2(indices[0], indices[1], indices[3], indices[2], name=self.name)
                )

        return tuple(tensors)


def _amplitude_as_uhf(
    amp: Tensor,
    type_uhf: type[Tensor],
    covariant: int,
    contravariant: int,
    target_rhf: bool = False,
) -> tuple[Base, ...]:
    """Convert a GHF amplitude tensor to a tuple of UHF tensors.

    Args:
        amp: GHF amplitude tensor.
        covariant: Number of covariant indices.
        contravariant: Number of contravariant indices.
        target_rhf: Whether the target is RHF tensors, which changes the desired
            antisymmetric format. If `True`, the output is still a tuple of UHF tensors, but these
            tensors can then be correctly converted to RHF tensors.

    Returns:
        Tuple of UHF tensors.
    """
    # Loop over spins
    tensors: list[Base] = []
    for spin_major in itertools.product("ab", repeat=max(covariant, contravariant)):
        for spin_minor in set(itertools.permutations(spin_major)):
            # Get the spin ordering
            if covariant > contravariant:
                spins = spin_major[:covariant] + spin_minor[:contravariant]
            else:
                spins = spin_minor[:contravariant] + spin_major[:covariant]

            # Check for fixed spins
            if any(index.spin and index.spin != spin for index, spin in zip(amp.indices, spins)):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index, spin in zip(amp.indices, spins))
            tensor_uhf = type_uhf(*indices, name=amp.name)

            # Check for RHF tensors
            if target_rhf:
                tensors.append(tensor_uhf)
                continue

            # Expand antisymmetry
            for perm_major in fully_antisymmetric_group(max(covariant, contravariant)).permutations:
                if covariant > contravariant:
                    perm = perm_major + Permutation(tuple(range(contravariant)), 1)
                else:
                    perm = Permutation(tuple(range(covariant)), 1) + perm_major
                spins_perm = tuple(spins[i] for i in perm.permutation)
                if spins == spins_perm:
                    tensors.append(tensor_uhf.permute_indices(perm))

    return tuple(tensors)


class T1(Tensor):
    """Class for the T1 amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 2:
            raise ValueError("T1 amplitude tensor must have two indices.")
        if name is None:
            name = "t1"
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # Loop over spins
        tensors = []
        for spin in ("a", "b"):
            # Check for fixed spins
            if any(index.spin and index.spin != spin for index in self.indices):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index in self.indices)
            tensors.append(uhf.T1(*indices, name=self.name))

        return tuple(tensors)


class T2(Tensor):
    """Class for the T2 amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 4:
            raise ValueError("T2 amplitude tensor must have four indices.")
        if name is None:
            name = "t2"
        if symmetry is None:
            symmetry = Symmetry(
                Permutation((0, 1, 2, 3), +1),
                Permutation((0, 1, 3, 2), -1),
                Permutation((1, 0, 2, 3), -1),
                Permutation((1, 0, 3, 2), +1),
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # TODO: Hardcode these for efficiency
        return _amplitude_as_uhf(self, uhf.T2, 2, 2, target_rhf=target_rhf)


class T3(Tensor):
    """Class for the T3 amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 6:
            raise ValueError("T3 amplitude tensor must have six indices.")
        if name is None:
            name = "t3"
        if symmetry is None:
            symmetry = Symmetry(
                *(
                    perm_bra + perm_ket
                    for perm_ket in fully_antisymmetric_group(3).permutations
                    for perm_bra in fully_antisymmetric_group(3).permutations
                )
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        return _amplitude_as_uhf(self, uhf.T3, 3, 3, target_rhf=target_rhf)


class L1(T1):
    """Class for the L1 amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 2:
            raise ValueError("L1 amplitude tensor must have two indices.")
        if name is None:
            name = "l1"
        T1.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # Loop over spins
        tensors = []
        for spin in ("a", "b"):
            # Check for fixed spins
            if any(index.spin and index.spin != spin for index in self.indices):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index in self.indices)
            tensors.append(uhf.L1(*indices))

        return tuple(tensors)


class L2(T2):
    """Class for the L2 amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 4:
            raise ValueError("L2 amplitude tensor must have four indices.")
        if name is None:
            name = "l2"
        T2.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # TODO: Hardcode these for efficiency
        return _amplitude_as_uhf(self, uhf.L2, 2, 2, target_rhf=target_rhf)


class L3(T3):
    """Class for the L3 amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 6:
            raise ValueError("L3 amplitude tensor must have six indices.")
        if name is None:
            name = "l3"
        T3.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # TODO: Hardcode these for efficiency
        return _amplitude_as_uhf(self, uhf.L3, 3, 3, target_rhf=target_rhf)


class R0(Tensor):
    """Class for the R0 amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 0:
            raise ValueError("R0 amplitude tensor must have zero indices.")
        if name is None:
            name = "r0"
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        return (uhf.R0(name=self.name),)


class R1ip(Tensor):
    """Class for the R1ip amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 1:
            raise ValueError("R1ip amplitude tensor must have two indices.")
        if name is None:
            name = "r1"
        if symmetry is None:
            symmetry = symmetric_group((0,))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # Loop over spins
        tensors = []
        for spin in ("a", "b"):
            # Check for fixed spins
            if any(index.spin and index.spin != spin for index in self.indices):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index in self.indices)
            tensors.append(uhf.R1ip(*indices, name=self.name))

        return tuple(tensors)


class R2ip(Tensor):
    """Class for the R2ip amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 3:
            raise ValueError("R2ip amplitude tensor must have two indices.")
        if name is None:
            name = "r2"
        if symmetry is None:
            symmetry = Symmetry(
                *(
                    perm_bra + perm_ket
                    for perm_ket in fully_antisymmetric_group(1).permutations
                    for perm_bra in fully_antisymmetric_group(2).permutations
                )
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        return _amplitude_as_uhf(self, uhf.R2ip, 2, 1, target_rhf=target_rhf)


class R3ip(Tensor):
    """Class for the R3ip amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 5:
            raise ValueError("R3ip amplitude tensor must have two indices.")
        if name is None:
            name = "r3"
        if symmetry is None:
            symmetry = Symmetry(
                *(
                    perm_bra + perm_ket
                    for perm_ket in fully_antisymmetric_group(2).permutations
                    for perm_bra in fully_antisymmetric_group(3).permutations
                )
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        return _amplitude_as_uhf(self, uhf.R3ip, 3, 2, target_rhf=target_rhf)


class R1ea(Tensor):
    """Class for the R1ea amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 1:
            raise ValueError("R1ea amplitude tensor must have two indices.")
        if name is None:
            name = "r1"
        if symmetry is None:
            symmetry = symmetric_group((0,))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        # Loop over spins
        tensors = []
        for spin in ("a", "b"):
            # Check for fixed spins
            if any(index.spin and index.spin != spin for index in self.indices):
                continue

            # Create the UHF tensor
            indices = tuple(index.copy(spin=spin) for index in self.indices)
            tensors.append(uhf.R1ea(*indices, name=self.name))

        return tuple(tensors)


class R2ea(Tensor):
    """Class for the R2ea amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 3:
            raise ValueError("R2ea amplitude tensor must have two indices.")
        if name is None:
            name = "r2"
        if symmetry is None:
            symmetry = Symmetry(
                *(
                    perm_bra + perm_ket
                    for perm_ket in fully_antisymmetric_group(2).permutations
                    for perm_bra in fully_antisymmetric_group(1).permutations
                )
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        return _amplitude_as_uhf(self, uhf.R2ea, 1, 2, target_rhf=target_rhf)


class R3ea(Tensor):
    """Class for the R3ea amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 5:
            raise ValueError("R3ea amplitude tensor must have two indices.")
        if name is None:
            name = "r3"
        if symmetry is None:
            symmetry = Symmetry(
                *(
                    perm_bra + perm_ket
                    for perm_ket in fully_antisymmetric_group(3).permutations
                    for perm_bra in fully_antisymmetric_group(2).permutations
                )
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        return _amplitude_as_uhf(self, uhf.R3ea, 2, 3, target_rhf=target_rhf)


class R1ee(T1):
    """Class for the R1ee amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 2:
            raise ValueError("R1ee amplitude tensor must have two indices.")
        if name is None:
            name = "r1"
        T1.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        return _amplitude_as_uhf(self, uhf.R1ee, 1, 1, target_rhf=target_rhf)


class R2ee(T2):
    """Class for the R2ee amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 4:
            raise ValueError("R2ee amplitude tensor must have four indices.")
        if name is None:
            name = "r2"
        T2.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        return _amplitude_as_uhf(self, uhf.R2ee, 2, 2, target_rhf=target_rhf)


class R3ee(T3):
    """Class for the R3ee amplitude tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 6:
            raise ValueError("R3ee amplitude tensor must have six indices.")
        if name is None:
            name = "r3"
        T3.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of tensors resulting from the conversion.
        """
        return _amplitude_as_uhf(self, uhf.R3ee, 3, 3, target_rhf=target_rhf)
