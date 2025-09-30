"""Class for algebraic operations."""

from __future__ import annotations

import itertools
from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING, TypedDict, TypeVar, cast

from albert import ALLOW_NON_EINSTEIN_NOTATION
from albert.base import Base
from albert.scalar import Scalar, _compose_scalar

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable

    from albert.index import Index
    from albert.tensor import Tensor, _TensorJSON

T = TypeVar("T", bound=Base)


def _check_indices(children: Iterable[Base]) -> None:
    """Check for Einstein notation.

    Args:
        children: Children to check.
    """
    counts: dict[Index, int] = defaultdict(int)
    for child in children:
        for index in child.internal_indices:
            counts[index] += 1
        for index in child.external_indices:
            counts[index] += 1
    if any(count > 2 for count in counts.values()) and not ALLOW_NON_EINSTEIN_NOTATION:
        raise ValueError(
            "Input arguments are not a valid Einstein notation.  Each index must appear at most "
            "twice."
        )


def _repr_brackets(obj: Base) -> str:
    """Helper function to add brackets to a string representation when needed.

    Args:
        obj: Object to represent.

    Returns:
        String representation.
    """
    if isinstance(obj, (Add, Mul)):
        return f"({obj})"
    return str(obj)


class _AlgebraicJSON(TypedDict):
    """Type for JSON representation of an algebraic operation."""

    _type: str
    _module: str
    children: tuple[_AlgebraicJSON | _TensorJSON, ...]


class Algebraic(Base):
    """Base class for algebraic operations.

    Args:
        children: Children to operate on.
    """

    __slots__ = ("_hash", "_children")

    _children: tuple[Base, ...]
    _compose: Callable[..., Base]

    def __init__(self, *children: Base):
        """Initialise the addition."""
        self._hash = None
        self._children = children

    def copy(self, *children: Base) -> Algebraic:
        """Return a copy of the object with optionally updated attributes."""
        if not children:
            children = self._children
        return self.__class__(*children)

    def map_indices(self, mapping: dict[Index, Index]) -> Algebraic:
        """Return a copy of the object with the indices mapped according to some dictionary.

        Args:
            mapping: map between old indices and new indices.

        Returns:
            Object with mapped indices.
        """
        children = [child.map_indices(mapping) for child in self._children]
        return self.copy(*children)

    def apply(
        self,
        function: Callable[[T], Base],
        node_type: type[T] | tuple[type[T], ...],
    ) -> Base:
        """Apply a function to nodes.

        Args:
            function: Functon to apply.
            node_type: Type of node to apply to.

        Returns:
            Object after applying function (if applicable).

        Note:
            The function is applied to the children first, and then to the object itself.
        """
        children = [child.apply(function, node_type) for child in self._children]
        if isinstance(self, node_type):
            return function(cast(T, self.copy(*children)))
        return self.copy(*children)

    def collect(self) -> Base:
        """Collect like terms in the top layer of the object.

        Returns:
            Object with like terms collected.
        """
        # Expand the object
        expanded = self.expand()

        # Collect like terms
        collected: dict[tuple[Base, ...], Scalar] = defaultdict(Scalar)
        for mul in expanded._children:
            parts: list[Base] = []
            factor = Scalar(1.0)
            if mul._children:
                for child in mul._children:
                    if isinstance(child, Scalar):
                        factor *= child
                    else:
                        parts.append(child)
            collected[tuple(parts)] += factor

        # Recompose the object
        expr = _compose_add(
            *[_compose_mul(factor, *tensors) for tensors, factor in collected.items()]
        )

        return expr

    def squeeze(self) -> Base:
        """Squeeze the object by removing any redundant algebraic operations.

        Returns:
            Object with redundant operations removed.
        """
        children = [child.squeeze() for child in self._children]
        if len(children) == 1:
            return children[0]
        else:
            return self.copy(*children)

    def canonicalise(self, indices: bool = False) -> Base:
        """Canonicalise the object.

        This function performs a greedy canonicalisation and may not always find the optimal
        canonical form. For a more exhaustive canonicalisation, use `canonicalise_exhaustive`.

        Args:
            indices: Whether to canonicalise the indices of the object. When `True`, this is
                performed for the outermost call in recursive calls.

        Returns:
            Object in canonical format.
        """
        children = sorted([child.canonicalise(indices=False) for child in self._children])
        expr = self._compose(*children)

        if indices:
            from albert.canon import canonicalise_indices

            expr = canonicalise_indices(expr)

        return expr.squeeze()

    def as_json(self) -> _AlgebraicJSON:
        """Return a JSON representation of the object.

        Returns:
            Object in JSON format.
        """
        return {
            "_type": self.__class__.__name__,
            "_module": self.__class__.__module__,
            "children": tuple(child.as_json() for child in self._children),
        }

    @classmethod
    def from_json(cls, data: _AlgebraicJSON) -> Algebraic:
        """Return an object loaded from a JSON representation.

        Returns:
            Object loaded from JSON representation.
        """
        children = [Base.from_json(child) for child in data["children"]]
        return cls(*children)

    @classmethod
    def from_sympy(cls, data: Any) -> Algebraic:
        """Return an object loaded from a sympy representation.

        Returns:
            Object loaded from sympy representation.
        """
        import sympy

        from albert.tensor import Tensor

        children: list[Base] = []
        for child in data.args:
            if isinstance(child, (sympy.Add, sympy.Mul)):
                children.extend(cls.from_sympy(child)._children)
            elif isinstance(child, sympy.Indexed):
                children.append(Tensor.from_sympy(child))
            elif isinstance(child, sympy.Float):
                children.append(_compose_scalar(child))
            else:
                raise ValueError(f"Unknown sympy object {child}.")

        return cls(*children)


def _compose_add(*children: Base, cls: type[Add] | None = None) -> Base:
    """Compose an addition. May not return an addition if unnecessary.

    Args:
        children: Children to add.
        cls: Class to use for addition.

    Returns:
        Composed object.
    """
    if cls is None:
        cls = Add
    if not issubclass(cls, Add):
        raise ValueError("cls must be a subclass of Add.")

    # If there are no arguments, it's zero
    if len(children) == 0:
        return Scalar(0.0)

    # Collect scalars
    scalar = Scalar(0.0)
    other: list[Base] = []
    for child in children:
        if isinstance(child, Scalar):
            scalar = Scalar(scalar._value + child._value)
        elif isinstance(child, Add):
            inner = _compose_add(*child._children)
            if inner._children:
                for child in inner._children:
                    if isinstance(child, Scalar):
                        scalar = Scalar(scalar._value + child._value)
                    else:
                        other.append(child)
        else:
            other.append(child)

    # If there are no other arguments, return the scalar
    if not other:
        return scalar

    # If the scalar is zero, don't include it
    if scalar._value == 0.0:
        if len(other) == 1:
            return other[0]
        return cls(*other)

    # Otherwise, include the scalar
    return cls(scalar, *other)


def _compose_mul(
    *children: Base, cls: type[Mul] | None = None, cls_scalar: type[Scalar] | None = None
) -> Base:
    """Compose a multiplication. May not return a multiplication if unnecessary.

    Args:
        children: Children to multiply.
        cls: Class to use for multiplication.
        cls_scalar: Class to use for scalars.

    Returns:
        Composed object.
    """
    if cls is None:
        cls = Mul
    if cls_scalar is None:
        cls_scalar = Scalar
    if not issubclass(cls, Mul):
        raise ValueError("cls must be a subclass of Mul.")
    if not issubclass(cls_scalar, Scalar):
        raise ValueError("cls_scalar must be a subclass of Scalar.")

    # If there are no arguments, it's one
    if len(children) == 0:
        return cls_scalar(1.0)

    # Collect scalars
    scalar = cls_scalar(1.0)
    other: list[Base] = []
    for child in children:
        if isinstance(child, Scalar):
            scalar = cls_scalar(scalar._value * child._value)
        elif isinstance(child, Mul):
            inner = _compose_mul(*child._children)
            if inner._children:
                for child in inner._children:
                    if isinstance(child, Scalar):
                        scalar = cls_scalar(scalar._value * child._value)
                    else:
                        other.append(child)
        else:
            other.append(child)

    # If the product of scalars is zero, return zero
    if scalar._value == 0.0:
        return cls_scalar(0.0)

    # If there are no other arguments, return the scalar
    if not other:
        return scalar

    # If the scalar is one, don't include it
    if scalar._value == 1.0:
        if len(other) == 1:
            return other[0]
        return cls(*other)

    # Otherwise, include the scalar
    return cls(scalar, *other)


class Add(Algebraic):
    """Class for an addition.

    Args:
        children: Children to add.
    """

    __slots__ = ("_hash", "_children")

    _score = 2
    _compose = staticmethod(_compose_add)

    def __init__(self, *children: Base):
        """Initialise the addition."""
        if len(set(tuple(sorted(child.external_indices)) for child in children)) > 1:
            raise ValueError("External indices in additions must be equal.")
        super().__init__(*children)

    @property
    def external_indices(self) -> tuple[Index, ...]:
        """Get the external indices (those that are not summed over)."""
        return self._children[0].external_indices  # Already checked on input

    @property
    def internal_indices(self) -> tuple[Index, ...]:
        """Get the internal indices (those that are summed over)."""
        return ()

    @property
    def disjoint(self) -> bool:
        """Return whether the object is disjoint."""
        return False

    def expand(self) -> ExpandedAddLayer:
        """Expand the object into the minimally nested format.

        Output has the form Add[Mul[Tensor | Scalar]].

        Returns:
            Object in expanded format.
        """
        # Recursively expand the parentheses
        children: list[ExpandedMulLayer] = []
        for child in self._children:
            children.extend(child.expand()._children)
        return ExpandedAddLayer(*children)

    def as_sympy(self) -> Any:
        """Return a sympy representation of the object.

        Returns:
            Object in sympy format.
        """
        return sum(child.as_sympy() for child in self._children)

    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        return " + ".join(map(_repr_brackets, self._children))

    def __add__(self, other: Base | float) -> Base:
        """Add two objects."""
        if isinstance(other, (int, float)):
            other = _compose_scalar(other)
        if isinstance(other, Add):
            return _compose_add(*self._children, *other._children)
        return _compose_add(*self._children, other)

    def __mul__(self, other: Base | float) -> Base:
        """Multiply two objects."""
        if isinstance(other, (int, float)):
            other = _compose_scalar(other)
        if isinstance(other, Mul):
            return _compose_mul(self, *other._children)
        return _compose_mul(self, other)


class Mul(Algebraic):
    """Class for an multiplication.

    Args:
        children: Children to multiply
    """

    __slots__ = ("_hash", "_children")

    _score = 3
    _compose = staticmethod(_compose_mul)

    def __init__(self, *children: Base):
        """Initialise the multiplication."""
        _check_indices(children)
        super().__init__(*children)

    @property
    def external_indices(self) -> tuple[Index, ...]:
        """Get the external indices (those that are not summed over)."""
        counts: dict[Index, int] = defaultdict(int)
        for child in self._children:
            for index in child.external_indices:
                counts[index] += 1
        return tuple(index for index, count in counts.items() if count == 1)

    @property
    def internal_indices(self) -> tuple[Index, ...]:
        """Get the internal indices (those that are summed over)."""
        counts: dict[Index, int] = defaultdict(int)
        for child in self._children:
            for index in child.external_indices:
                counts[index] += 1
        return tuple(index for index, count in counts.items() if count > 1)

    @property
    def disjoint(self) -> bool:
        """Return whether the object is disjoint."""
        return len(self.internal_indices) == 0

    def expand(self) -> ExpandedAddLayer:
        """Expand the object into the minimally nested format.

        Output has the form Add[Mul[Tensor | Scalar]].

        Returns:
            Object in expanded format.
        """
        # Recursively expand the parentheses
        children: list[ExpandedMulLayer] = []
        for child in self._children:
            inner_children: tuple[ExpandedMulLayer, ...] = child.expand()._children
            if not children:
                children.extend(list(inner_children))
            else:
                children = [
                    cast(ExpandedMulLayer, a * b)
                    for a, b in itertools.product(children, inner_children)
                ]
        return ExpandedAddLayer(*children)

    def as_sympy(self) -> Any:
        """Return a sympy representation of the object.

        Returns:
            Object in sympy format.
        """
        return reduce(lambda x, y: x * y, (child.as_sympy() for child in self._children))

    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        return " * ".join(map(_repr_brackets, self._children))

    def __add__(self, other: Base | float) -> Base:
        """Add two objects."""
        if isinstance(other, (int, float)):
            other = _compose_scalar(other)
        if isinstance(other, Add):
            return _compose_add(self, *other._children)
        return _compose_add(self, other)

    def __mul__(self, other: Base | float) -> Base:
        """Multiply two objects."""
        if isinstance(other, (int, float)):
            other = _compose_scalar(other)
        if isinstance(other, Mul):
            return _compose_mul(*self._children, *other._children)
        return _compose_mul(*self._children, other)


class ExpandedMulLayer(Mul):
    """`Mul` layer of the result of `Base.expand`.

    The `Base.expand` method returns and object that gurantees the structure
    Add[Mul[Tensor | Scalar]]. This is used as a type hint for the return type.
    """

    _children: tuple[Tensor | Scalar, ...]


class ExpandedAddLayer(Add):
    """`Add` layer of the result of `Base.expand`.

    The `Base.expand` method returns and object that gurantees the structure
    Add[Mul[Tensor | Scalar]]. This is used as a type hint for the return type.
    """

    _children: tuple[ExpandedMulLayer, ...]
