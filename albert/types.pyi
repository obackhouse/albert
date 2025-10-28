from __future__ import annotations
from typing import Any, Hashable, Protocol, Optional, TypedDict


class Comparable(Protocol):
    """Protocol for comparable objects."""

    def __eq__(self, other: Any) -> bool:
        """Check if two objects are equal."""
        pass

    def __lt__(self, other: Any) -> bool:
        """Check if an object precedes another."""
        pass


class SerialisedField(Comparable, Hashable, Protocol):
    """Protocol for the fields of serialised formats."""

    pass


class _ScalarJSON(TypedDict):
    """Type for JSON representation of a scalar."""

    _type: str
    _module: str
    value: float


class _IndexJSON(TypedDict):
    """Type for JSON representation of an index."""

    _type: str
    _module: str
    name: str
    spin: Optional[str]
    space: Optional[str]


class _PermutationJSON(TypedDict):
    """Type for JSON representation of a permutation."""

    _type: str
    _module: str
    permutation: tuple[int, ...]
    sign: int


class _SymmetryJSON(TypedDict):
    """Type for JSON representation of a symmetry group."""

    _type: str
    _module: str
    permutations: tuple[_PermutationJSON, ...]


class _TensorJSON(TypedDict):
    """Type for JSON representation of a tensor."""

    _type: str
    _module: str
    indices: tuple[_IndexJSON, ...]
    name: str
    symmetry: Optional[_SymmetryJSON]


class _AlgebraicJSON(TypedDict):
    """Type for JSON representation of an algebraic operation."""

    _type: str
    _module: str
    children: tuple[_AlgebraicJSON | _TensorJSON, ...]


class _ExpressionJSON(TypedDict):
    """Type for JSON representation of an expression."""

    _type: str
    _module: str
    lhs: _TensorJSON
    rhs: _TensorJSON | _AlgebraicJSON
