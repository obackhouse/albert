"""Classes for tensors for quantum chemistry."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from albert.base import Base
from albert.tensor import Tensor

if TYPE_CHECKING:
    pass


class SpinMixin:
    """Mixin class to expose spin conversion methods."""

    @abstractmethod
    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of expressions resulting from the conversion.
        """
        pass

    @abstractmethod
    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Expression resulting from the conversion.
        """
        pass


class QTensor(Tensor, SpinMixin):
    """Class for a tensor in quantum chemistry.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
    """

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of expressions resulting from the conversion.
        """
        raise NotImplementedError(f"as_uhf not implemented for tensor {self.name}.")

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Expression resulting from the conversion.
        """
        raise NotImplementedError(f"as_rhf not implemented for tensor {self.name}.")
