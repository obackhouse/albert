"""Classes for RHF tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from albert.symmetry import Permutation, Symmetry, symmetric_group
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
            name = "t"
        if symmetry is None:
            symmetry = symmetric_group((0, 1))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "t"
        if symmetry is None:
            symmetry = Symmetry(
                Permutation((0, 1, 2, 3), +1),
                Permutation((1, 0, 3, 2), +1),
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "t"
        if symmetry is None:
            symmetry = Symmetry(
                Permutation((0, 1, 2, 3, 4, 5), +1),
                Permutation((0, 1, 2, 5, 4, 3), -1),
                Permutation((2, 1, 0, 3, 4, 5), -1),
                Permutation((2, 1, 0, 5, 4, 3), +1),
            )
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "l"
        T1.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "l"
        T2.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "l"
        T3.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "r"
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "r"
        if symmetry is None:
            symmetry = symmetric_group((0,))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "r"
        if symmetry is None:
            symmetry = symmetric_group((0, 1, 2))  # FIXME?
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "r"
        if symmetry is None:
            symmetry = symmetric_group((0, 1, 2, 3, 4))  # FIXME?
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "r"
        if symmetry is None:
            symmetry = symmetric_group((0,))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "r"
        if symmetry is None:
            symmetry = symmetric_group((0, 1, 2))  # FIXME?
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "r"
        if symmetry is None:
            symmetry = symmetric_group((0, 1, 2, 3, 4))  # FIXME?
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "r"
        T1.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "r"
        T2.__init__(self, *indices, name=name, symmetry=symmetry)


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
            name = "r"
        T3.__init__(self, *indices, name=name, symmetry=symmetry)
