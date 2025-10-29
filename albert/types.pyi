from __future__ import annotations
from typing import Any, Hashable, Protocol, Optional, TypedDict


EvaluatorArrayDict = dict[str, dict[tuple[str, ...] | str, Any]] | dict[str, Any]


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


class _BaseJSON(TypedDict):
    """Base type for JSON representation."""

    _type: str
    _module: str


class _ScalarJSON(_BaseJSON):
    """Type for JSON representation of a scalar."""

    value: float


class _IndexJSON(_BaseJSON):
    """Type for JSON representation of an index."""

    name: str
    spin: Optional[str]
    space: Optional[str]


class _PermutationJSON(_BaseJSON):
    """Type for JSON representation of a permutation."""

    permutation: tuple[int, ...]
    sign: int


class _SymmetryJSON(_BaseJSON):
    """Type for JSON representation of a symmetry group."""

    permutations: tuple[_PermutationJSON, ...]


class _TensorJSON(_BaseJSON):
    """Type for JSON representation of a tensor."""

    indices: tuple[_IndexJSON, ...]
    name: str
    symmetry: Optional[_SymmetryJSON]


class _AlgebraicJSON(_BaseJSON):
    """Type for JSON representation of an algebraic operation."""

    children: tuple[_AlgebraicJSON | _TensorJSON, ...]


class _ExpressionJSON(_BaseJSON):
    """Type for JSON representation of an expression."""

    lhs: _TensorJSON
    rhs: _TensorJSON | _AlgebraicJSON
