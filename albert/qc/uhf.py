"""Classes for UHF tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from albert.base import Base
from albert.qc import rhf
from albert.scalar import Scalar
from albert.symmetry import Permutation, Symmetry, fully_antisymmetric_group, symmetric_group
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Optional

    from albert.index import Index


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        indices = tuple(index.copy(spin="r") for index in self.indices)
        return rhf.Fock(*indices, name=self.name)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        indices = tuple(index.copy(spin="r") for index in self.indices)
        return rhf.RDM1(*indices, name=self.name)


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
            name = "d"
        if symmetry is None:
            symmetry = symmetric_group((0, 1), (1, 0))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        indices = tuple(index.copy(spin="r") for index in self.indices)
        return rhf.Delta(*indices, name=self.name)


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
            symmetry = symmetric_group(
                (0, 1, 2, 3),
                (0, 1, 3, 2),
                (1, 0, 2, 3),
                (1, 0, 3, 2),
                (2, 3, 0, 1),
                (3, 2, 0, 1),
                (2, 3, 1, 0),
                (3, 2, 1, 0),
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        indices = tuple(index.copy(spin="r") for index in self.indices)
        return rhf.ERI(*indices, name=self.name)


class CDERI(Tensor):
    """Class for the CDERI tensor.

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
            raise ValueError("CDERI tensor must have four indices.")
        if name is None:
            name = "v"
        if symmetry is None:
            symmetry = symmetric_group((0, 1, 2), (0, 2, 1))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        indices = tuple(index.copy(spin="r") for index in self.indices)
        return rhf.CDERI(*indices, name=self.name)


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
            raise ValueError("ERI tensor must have four indices.")
        if name is None:
            name = "Γ"
        if symmetry is None:
            symmetry = symmetric_group((0, 1, 2, 3))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        indices = tuple(index.copy(spin="r") for index in self.indices)
        return rhf.RDM2(*indices, name=self.name)


def _amplitude_as_rhf(
    amp: Tensor,
    type_rhf: type[Tensor],
    covariant: int,
    contravariant: int,
) -> Base:
    """Convert a UHF amplitude tensor to a RHF tensor.

    Args:
        amp: UHF amplitude tensor.
        covariant: Number of covariant indices.
        contravariant: Number of contravariant indices.

    Returns:
        RHF tensor.
    """
    # Spin flip if necessary
    na = sum(index.spin == "a" for index in amp.external_indices)
    nb = sum(index.spin == "b" for index in amp.external_indices)
    if nb > na:
        indices = tuple(index.spin_flip() for index in amp.external_indices)
        amp = amp.copy(*indices)

    # Expand same spin amplitudes as linear combination of mixed spin amplitudes
    amps: list[Base] = [amp]
    if amp.rank > 2:
        if all(index.spin == "a" for index in amp.external_indices):
            # Get the mixed spin amplitudes
            amps = []
            for k in range(covariant):
                spin = [("a", "b")[j % 2] for j in range(covariant)]
                spin += [("a", "b")[j == k] for j in range(contravariant)]
                indices = tuple(index.copy(spin=s) for index, s in zip(amp.external_indices, spin))
                amps.append(amp.copy(*indices))

    # Canonicalise the amplitudes
    amps = [amp.canonicalise() for amp in amps]

    # Relabel the indices
    for i, amp in enumerate(amps):
        indices = tuple(index.copy(spin="r") for index in amp.external_indices)
        amps[i] = amp.apply(
            lambda tensor: type_rhf(*indices, name=tensor.name), node_type=Tensor  # noqa: B023
        )

    return sum(amps, Scalar(0.0))


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
        if symmetry is None:
            symmetry = symmetric_group((0, 1))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        indices = tuple(index.copy(spin="r") for index in self.indices)
        return rhf.T1(*indices, name=self.name)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.T2, 2, 2)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.T3, 3, 3)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        indices = tuple(index.copy(spin="r") for index in self.indices)
        return rhf.L1(*indices, name=self.name)


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

    def as_rhf(self) -> Base:
        """Conver the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.L2, 2, 2)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.L3, 3, 3)


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

    def as_rhf(self) -> Base:
        """Convert the indices without spin to indices with spin.

        Returns:
            Tensor resulting from the conversion.
        """
        return rhf.R0(name=self.name)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.R1ip, 1, 0)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.R2ip, 2, 1)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.R3ip, 3, 2)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.R1ea, 0, 1)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.R2ea, 1, 2)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.R3ea, 2, 3)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.R1ee, 1, 1)


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

    def as_rhf(self) -> Base:
        """Conver the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.R2ee, 2, 2)


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

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Tensor resulting from the conversion.
        """
        return _amplitude_as_rhf(self, rhf.R3ee, 3, 3)
