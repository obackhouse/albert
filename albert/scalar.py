"""Classes for scalars."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from albert.base import _INTERN_TABLE, Base

if TYPE_CHECKING:
    from typing import Any, Optional

    from albert.index import Index
    from albert.types import _ScalarJSON

T = TypeVar("T", bound=Base)

_ZERO = 1e-12


class Scalar(Base):
    """Class for a scalar.

    Args:
        value: Value of the scalar.
    """

    __slots__ = ("_value", "_hash", "_children", "_internal_indices", "_external_indices")

    _score = 1

    def __init__(self, value: float = 0.0):
        """Initialise the tensor."""
        self._value = value
        self._hash = None
        self._children = None
        self._internal_indices = ()
        self._external_indices = ()

    @classmethod
    def factory(cls: type[Scalar], value: float) -> Scalar:
        """Factory method to create a new object.

        Args:
            value: Value of the scalar.

        Returns:
            Algebraic object. In general, `factory` methods may return objects of a different type
            to the class they are called on.
        """
        if not issubclass(cls, Scalar):
            raise TypeError(f"cls must be a subclass of Scalar, got {cls}")

        # Perform some basic simplifications
        if abs(value) < _ZERO:
            value = 0.0
        elif abs(value % 1) < _ZERO:
            value = int(value)

        # Build a key for interning
        key = (cls, value)

        def create() -> Scalar:
            return cls(value)

        return cast(Scalar, _INTERN_TABLE.get(key, create))

    @property
    def value(self) -> float:
        """Get the value of the scalar."""
        return self._value

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
            other = self.factory(other)
        if isinstance(other, Scalar):
            return self.factory(self.value + other.value)
        return NotImplemented

    def __mul__(self, other: Base | float) -> Scalar:
        """Multiply two objects."""
        if isinstance(other, (int, float)):
            other = self.factory(other)
        if isinstance(other, Scalar):
            return self.factory(self.value * other.value)
        return NotImplemented
