"""Classes for tensor expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from albert.base import Base, Serialisable
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable

    from albert.index import Index
    from albert.types import EvaluatorArrayDict, SerialisedField, _ExpressionJSON


class Expression(Serialisable):
    """Class for a tensor expression.

    Args:
        lhs: The left-hand side of the expression (a tensor).
        rhs: The right-hand side of the expression.
    """

    _lhs: Tensor
    _rhs: Base

    def __init__(self, lhs: Tensor, rhs: Base) -> None:
        """Initialise the expression."""
        if set(lhs.indices) != set(rhs.external_indices):
            raise ValueError("LHS indices must match RHS external indices.")
        self._hash = None
        self._lhs = lhs
        self._rhs = rhs

    @property
    def lhs(self) -> Tensor:
        """Get the left-hand side of the expression."""
        return self._lhs

    @property
    def rhs(self) -> Base:
        """Get the right-hand side of the expression."""
        return self._rhs

    @property
    def external_indices(self) -> tuple[Index, ...]:
        """Get the external indices of the expression."""
        return self._rhs.external_indices

    @property
    def internal_indices(self) -> tuple[Index, ...]:
        """Get the internal indices of the expression."""
        return self._rhs.internal_indices

    def expand(self) -> Expression:
        """Expand the RHS into the minimally nested format.

        Output RHS has the form Add[Mul[Tensor | Scalar]].

        Returns:
            Object in expanded format.
        """
        return Expression(self.lhs, self.rhs.expand())

    def copy(self) -> Expression:
        """Return a copy of the object.

        Returns:
            Copy of the object.
        """
        return Expression(self._lhs.copy(), self._rhs.copy())

    def evaluate(
        self,
        arrays: EvaluatorArrayDict,
        einsum: Callable[..., Any],
    ) -> Any:
        """Evaluate the node numerically.

        Args:
            arrays: Mapping to provide numerical arrays for tensors. The mapping must be in one of
                the following formats:
                    1. ``{tensor_name: { (space1, space2, ...): array, ... }, ...}``
                    2. ``{tensor_name: { "space1space2...": array, ...}, ...}``
                    3. ``{tensor_name: array, ...}`` (only for tensors with no indices)
            einsum: Function to perform tensor contraction.

        Returns:
            Evaluated node, as an array.
        """
        return self.rhs.evaluate(arrays, einsum)

    def as_json(self) -> _ExpressionJSON:
        """Return a JSON representation of the object.

        Returns:
            Object in JSON format.
        """
        return {
            "_type": self.__class__.__name__,
            "_module": self.__class__.__module__,
            "lhs": self._lhs.as_json(),
            "rhs": self._rhs.as_json(),
        }

    @classmethod
    def from_json(cls, data: _ExpressionJSON) -> Expression:
        """Return an object loaded from a JSON representation.

        Returns:
            Object loaded from JSON representation.
        """
        return cls(Tensor.from_json(data["lhs"]), Base.from_json(data["rhs"]))

    def _hashable_fields(self) -> Iterable[SerialisedField]:
        """Yield fields of the hashable representation."""
        yield self._lhs
        yield self._rhs

    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        return f"{self._lhs} = {self._rhs}"
