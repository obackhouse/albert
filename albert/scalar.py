"""Classes for scalars."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from albert.base import Base

if TYPE_CHECKING:
    from typing import Any, Optional

    from albert.index import Index
    from albert.types import _ScalarJSON

T = TypeVar("T", bound=Base)

_ZERO = 1e-12


def _compose_scalar(value: float) -> Scalar:
    """Compose a scalar."""
    if abs(value) < _ZERO:
        return Scalar(0)
    if abs(value % 1) < _ZERO:
        return Scalar(int(value))
    return Scalar(value)


class Scalar(Base):
    """Class for a scalar.

    Args:
        value: Value of the scalar.
    """

    _score = 1

    def __init__(self, value: float = 0.0):
        """Initialise the tensor."""
        self._value = value
        self._hash = None
        self._children = None

    @property
    def value(self) -> float:
        """Get the value of the scalar."""
        return self._value

    @property
    def external_indices(self) -> tuple[Index, ...]:
        """Get the external indices (those that are not summed over)."""
        return ()

    @property
    def internal_indices(self) -> tuple[Index, ...]:
        """Get the internal indices (those that are summed over)."""
        return ()

    @property
    def disjoint(self) -> bool:
        """Return whether the object is disjoint."""
        return False

    def copy(self, value: Optional[float] = None) -> Scalar:
        """Return a copy of the object with optionally updated attributes.

        Args:
            value: New value.

        Returns:
            Copy of the object.
        """
        if value is None:
            value = self.value
        return Scalar(value)

    def map_indices(self, mapping: dict[Index, Index]) -> Scalar:
        """Return a copy of the object with the indices mapped according to some dictionary.

        Args:
            mapping: map between old indices and new indices.

        Returns:
            Object with mapped indices.
        """
        return self

    def canonicalise(self, indices: bool = False) -> Scalar:
        """Canonicalise the object.

        The results of this function for equivalent representations should be equal.

        Args:
            indices: Whether to canonicalise the indices of the object. When `True`, this is
                performed for the outermost call in recursive calls.

        Returns:
            Object in canonical format.
        """
        return self

    def expand(self) -> Base:
        """Expand the object into the minimally nested format.

        Output has the form Add[Mul[Tensor | Scalar]].

        Returns:
            Object in expanded format.
        """
        from albert.algebra import Add, Mul  # FIXME

        return Add(Mul(self))

    def collect(self) -> Scalar:
        """Collect like terms in the top layer of the object.

        Returns:
            Object with like terms collected.
        """
        return self

    def squeeze(self) -> Base:
        """Squeeze the object by removing any redundant algebraic operations.

        Returns:
            Object with redundant operations removed.
        """
        return self

    def as_sympy(self) -> Any:
        """Return a sympy representation of the object.

        Returns:
            Object in sympy format.
        """
        import sympy

        return sympy.Float(self.value)

    @classmethod
    def from_sympy(cls, data: Any) -> Scalar:
        """Return an object loaded from a sympy representation.

        Returns:
            Object loaded from sympy representation.
        """
        return cls(float(data))

    def as_json(self) -> _ScalarJSON:
        """Return a JSON representation of the object.

        Returns:
            Object in JSON format.
        """
        return {
            "_type": self.__class__.__name__,
            "_module": self.__class__.__module__,
            "value": self.value,
        }

    @classmethod
    def from_json(cls, data: _ScalarJSON) -> Scalar:
        """Return an object loaded from a JSON representation.

        Returns:
            Object loaded from JSON representation.
        """
        return cls(data["value"])

    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        body = str(self.value)
        while body.endswith("0") and "." in body:
            body = body[:-1]
        body = body.rstrip(".")
        return body

    def __add__(self, other: Base | float) -> Scalar:
        """Add two objects."""
        if isinstance(other, (int, float)):
            other = _compose_scalar(other)
        if isinstance(other, Scalar):
            return _compose_scalar(self.value + other.value)
        return NotImplemented

    def __mul__(self, other: Base | float) -> Scalar:
        """Multiply two objects."""
        if isinstance(other, (int, float)):
            other = _compose_scalar(other)
        if isinstance(other, Scalar):
            return _compose_scalar(self.value * other.value)
        return NotImplemented
