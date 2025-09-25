"""Classes for tensor expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from albert.base import Base, Serialisable, SerialisedField
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Iterable

    from albert.algebra import _AlgebraicJSON
    from albert.index import Index
    from albert.tensor import _TensorJSON


class _ExpressionJSON(TypedDict):
    """Type for JSON representation of an expression."""

    _type: str
    _module: str
    lhs: _TensorJSON
    rhs: _TensorJSON | _AlgebraicJSON


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
