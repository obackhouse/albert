"""Class for algebraic operations."""

from __future__ import annotations

import itertools
from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING, TypeVar

from albert import ALLOW_NON_EINSTEIN_NOTATION
from albert.base import Base, _INTERN_TABLE
from albert.scalar import Scalar

if TYPE_CHECKING:
    from typing import Any, Iterable

    from albert.index import Index
    from albert.types import _AlgebraicJSON

T = TypeVar("T", bound=Base)


def _check_indices(children: Iterable[Base]) -> dict[Index, int]:
    """Check for Einstein notation.

    Args:
        children: Children to check.

    Returns:
        Dictionary with counts of each index.

    Raises:
        ValueError: If the indices do not satisfy Einstein notation.
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
    return counts


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


class Algebraic(Base):
    """Base class for algebraic operations.

    Args:
        children: Children to operate on.
    """

    __slots__ = ("_hash", "_children")

    _children: tuple[Base, ...]

    def __init__(self, *children: Base):
        """Initialise the addition."""
        self._hash = None
        self._children = children

    def copy(self, *children: Base) -> Algebraic:
        """Return a copy of the object with optionally updated attributes."""
        if not children:
            children = self.children
        return self.__class__(*children)

    def map_indices(self, mapping: dict[Index, Index]) -> Algebraic:
        """Return a copy of the object with the indices mapped according to some dictionary.

        Args:
            mapping: map between old indices and new indices.

        Returns:
            Object with mapped indices.
        """
        children = [child.map_indices(mapping) for child in self.children]
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
        for mul in expanded.children:
            parts: list[Base] = []
            factor = Scalar.factory(1.0)
            if mul.children:
                for child in mul.children:
                    if isinstance(child, Scalar):
                        factor *= child
                    else:
                        parts.append(child)
            collected[tuple(parts)] += factor

        # Recompose the object
        expr = Add.factory(
            *[Mul.factory(factor, *tensors) for tensors, factor in collected.items()]
        )

        return expr

    def squeeze(self) -> Base:
        """Squeeze the object by removing any redundant algebraic operations.

        Returns:
            Object with redundant operations removed.
        """
        children = [child.squeeze() for child in self.children]
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
        children = sorted([child.canonicalise(indices=False) for child in self.children])
        expr = self.factory(*children)

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
            "children": tuple(child.as_json() for child in self.children),
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
                children.extend(cls.from_sympy(child).children)
            elif isinstance(child, sympy.Indexed):
                children.append(Tensor.from_sympy(child))
            elif isinstance(child, sympy.Float):
                children.append(Scalar.factory(child))
            else:
                raise ValueError(f"Unknown sympy object {child}.")

        return cls(*children)


class Add(Algebraic):
    """Class for an addition.

    Args:
        children: Children to add.
    """

    __slots__ = ("_hash", "_children", "_internal_indices", "_external_indices")

    _score = 2

    def __init__(self, *children: Base):
        """Initialise the addition."""
        if len(set(tuple(sorted(child.external_indices)) for child in children)) > 1:
            raise ValueError("External indices in additions must be equal.")
        super().__init__(*children)

        # Precompute indices
        self._external_indices = children[0].external_indices
        self._internal_indices = ()

    @classmethod
    def factory(cls: type[Add], *children: Base, cls_scalar: type[Scalar] | None = None) -> Base:
        """Factory method to create a new object.

        Args:
            cls: The class of the addition to create.
            children: The children of the addition.
            cls_scalar: Class to use for scalars.

        Returns:
            Algebraic object. In general, `factory` methods may return objects of a different type
            to the class they are called on.
        """
        if cls_scalar is None:
            cls_scalar = Scalar
        if not issubclass(cls, Add):
            raise TypeError(f"cls must be a subclass of Add, got {cls}")
        if not issubclass(cls_scalar, Scalar):
            raise TypeError(f"cls_scalar must be a subclass of Scalar, got {cls_scalar}")

        # Perform some basic simplifications
        if len(children) == 0:
            return cls_scalar.factory(0.0)
        value = 0.0
        other: list[Base] = []
        for child in children:
            if isinstance(child, Scalar):
                value += child.value
            elif isinstance(child, Add):
                inner = Add.factory(*child.children)
                if inner.children:
                    for grandchild in inner.children:
                        if isinstance(grandchild, Scalar):
                            value += grandchild.value
                        else:
                            other.append(grandchild)
            else:
                other.append(child)

        # Build a key for interning
        key = (cls, value, tuple(other))  # Commutative but not canonical

        def create() -> Base:
            if not other:
                return cls(cls_scalar.factory(value))
            if abs(value) < 1e-12:
                if len(other) == 1:
                    return other[0]
                return cls(*other)
            return cls(cls_scalar.factory(value), *other)

        return _INTERN_TABLE.get(key, create)

    @property
    def disjoint(self) -> bool:
        """Return whether the object is disjoint."""
        return False

    def expand(self) -> Base:
        """Expand the object into the minimally nested format.

        Output has the form `Add[Mul[Tensor | Scalar]]`.

        Returns:
            Object in expanded format.
        """
        # Recursively expand the parentheses
        children: list[Base] = []
        for child in self.children:
            children.extend(child.expand().children)
        return Add(*children)

    def as_sympy(self) -> Any:
        """Return a sympy representation of the object.

        Returns:
            Object in sympy format.
        """
        return sum(child.as_sympy() for child in self.children)

    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        return " + ".join(map(_repr_brackets, self.children))

    def __add__(self, other: Base | float) -> Base:
        """Add two objects."""
        if isinstance(other, (int, float)):
            other = Scalar.factory(other)
        if isinstance(other, Add):
            return Add.factory(*self._children, *other.children)
        return Add.factory(*self.children, other)

    def __mul__(self, other: Base | float) -> Base:
        """Multiply two objects."""
        if isinstance(other, (int, float)):
            other = Scalar.factory(other)
        if isinstance(other, Mul):
            return Mul.factory(self, *other.children)
        return Mul.factory(self, other)


class Mul(Algebraic):
    """Class for an multiplication.

    Args:
        children: Children to multiply
    """

    __slots__ = ("_hash", "_children", "_internal_indices", "_external_indices")

    _score = 3

    def __init__(self, *children: Base):
        """Initialise the multiplication."""
        super().__init__(*children)

        # Precompute indices
        counts = _check_indices(children)
        self._external_indices = tuple(index for index, count in counts.items() if count == 1)
        self._internal_indices = tuple(index for index, count in counts.items() if count > 1)

    @classmethod
    def factory(cls: type[Mul], *children: Base, cls_scalar: type[Scalar] | None = None) -> Base:
        """Factory method to create a new object.

        Args:
            cls: The class of the multiplication to create.
            children: The children of the multiplication.
            cls_scalar: Class to use for scalars.

        Returns:
            Algebraic object. In general, `factory` methods may return objects of a different type
            to the class they are called on.
        """
        if cls_scalar is None:
            cls_scalar = Scalar
        if not issubclass(cls, Mul):
            raise TypeError(f"cls must be a subclass of Mul, got {cls}")
        if not issubclass(cls_scalar, Scalar):
            raise TypeError(f"cls_scalar must be a subclass of Scalar, got {cls_scalar}")

        # Perform some basic simplifications
        if len(children) == 0:
            return cls_scalar.factory(1.0)
        value = 1.0
        other: list[Base] = []
        for child in children:
            if isinstance(child, Scalar):
                value *= child.value
            elif isinstance(child, Mul):
                inner = Mul.factory(*child.children)
                if inner.children:
                    for grandchild in inner.children:
                        if isinstance(grandchild, Scalar):
                            value *= grandchild.value
                        else:
                            other.append(grandchild)
            else:
                other.append(child)

        # If the product of scalars is zero, return zero
        if value == 0.0:
            return cls_scalar.factory(0.0)

        # Build a key for interning
        key = (cls, value, tuple(other))  # Commutative but not canonical

        def create() -> Base:
            if not other:
                return cls(cls_scalar.factory(value))
            if abs(value - 1.0) < 1e-12:
                if len(other) == 1:
                    return other[0]
                return cls(*other)
            return cls(cls_scalar.factory(value), *other)

        return _INTERN_TABLE.get(key, create)

    @property
    def disjoint(self) -> bool:
        """Return whether the object is disjoint."""
        return len(self.internal_indices) == 0

    def expand(self) -> Base:
        """Expand the object into the minimally nested format.

        Output has the form Add[Mul[Tensor | Scalar]].

        Returns:
            Object in expanded format.
        """
        # Recursively expand the parentheses
        children: list[Base] = []
        for child in self.children:
            inner_children: tuple[Base, ...] = child.expand().children
            if not children:
                children.extend(list(inner_children))
            else:
                children = [a * b for a, b in itertools.product(children, inner_children)]
        return Add(*children)

    def as_sympy(self) -> Any:
        """Return a sympy representation of the object.

        Returns:
            Object in sympy format.
        """
        return reduce(lambda x, y: x * y, (child.as_sympy() for child in self.children))

    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        return " * ".join(map(_repr_brackets, self.children))

    def __add__(self, other: Base | float) -> Base:
        """Add two objects."""
        if isinstance(other, (int, float)):
            other = Scalar.factory(other)
        if isinstance(other, Add):
            return Add.factory(self, *other.children)
        return Add.factory(self, other)

    def __mul__(self, other: Base | float) -> Base:
        """Multiply two objects."""
        if isinstance(other, (int, float)):
            other = Scalar.factory(other)
        if isinstance(other, Mul):
            return Mul.factory(*self.children, *other.children)
        return Mul.factory(*self.children, other)
