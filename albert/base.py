"""Base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Optional, TypeVar

    from albert.algebra import ExpandedAddLayer
    from albert.index import Index
    from albert.symmetry import Permutation
    from albert.types import SerialisedField

    T = TypeVar("T", bound="Base")


class Serialisable(ABC):
    """Base class for serialisable objects."""

    _hash: Optional[int]

    @abstractmethod
    def as_json(self) -> Any:
        """Return a JSON representation of the object.

        Returns:
            Object in JSON format.
        """
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: Any) -> Serialisable:
        """Return an object loaded from a JSON representation.

        Returns:
            Object loaded from JSON representation.
        """
        pass

    def __hash__(self) -> int:
        """Return the hash of the object.

        Returns:
            The integer hash of the object.

        Note:
            By default, all `Serialisable` objects are hashable. Subclasses of `Serialisable`
            should implement the `_hashable_fields` method to define the fields in the hashable
            representation. The hash is computed lazily and stored in the `_hash` attribute.
        """
        if self._hash is None:
            self._hash = hash(self._hashable())
        return self._hash

    def _hashable(self) -> tuple[SerialisedField, ...]:
        """Return a hashable representation."""
        return tuple(self._hashable_fields())

    @abstractmethod
    def _hashable_fields(self) -> Iterable[SerialisedField]:
        """Yield fields of the hashable representation."""
        pass

    def __eq__(self, other: Any) -> bool:
        """Check if two objects are equal."""
        if self is other:
            return True
        if not isinstance(other, Serialisable):
            return False
        for a, b in zip(self._hashable_fields(), other._hashable_fields()):
            if a != b:
                return False
        return True

    def __lt__(self, other: Serialisable) -> bool:
        """Check if an object precedes another."""
        if self is other:
            return False
        for a, b in zip(self._hashable_fields(), other._hashable_fields()):
            if a != b:
                return bool(a < b)
        return False


def _sign_penalty(base: Base) -> int:
    """Return a penalty for the sign in scalars in a base object.

    Args:
        base: Base object to check.

    Returns:
        Penalty for the sign.
    """
    if not base._children:
        return 0
    penalty = 1
    if base._children:
        for child in base._children:
            if hasattr(child, "_value"):
                penalty *= 1 if child._value < 0 else -1
    return penalty


class Base(Serialisable):
    """Base class for algebraic types."""

    _score: int
    _children: Optional[tuple[Base, ...]]
    _penalties: tuple[Callable[[Base], int], ...] = (_sign_penalty,)

    def search_leaves(self, type_filter: type[T]) -> Iterable[T]:
        """Depth-first search through leaves.

        Args:
            type_filter: Type to filter by.

        Yields:
            Elements of the `_children` tuple, recursively.
        """
        if self._children is not None:
            for child in self._children:
                yield from child.search_leaves(type_filter)
        elif isinstance(self, type_filter):
            yield self

    def search_nodes(self, type_filter: type[T]) -> Iterable[T]:
        """Depth-first search through nodes.

        Args:
            type_filter: Type to filter by.

        Yields:
            Nodes and elements of the `_children` tuple, recursively.
        """
        if isinstance(self, type_filter):
            yield self
        if self._children is not None:
            for child in self._children:
                yield from child.search_nodes(type_filter)

    def search_children(self, type_filter: type[T]) -> Iterable[T]:
        """Search through direct children.

        Args:
            type_filter: Type to filter by.

        Yields:
            Elements of the `_children` tuple.
        """
        if self._children is not None:
            for child in self._children:
                if isinstance(child, type_filter):
                    yield child

    @property
    def is_leaf(self) -> bool:
        """Get whether the object is a leaf in a tree."""
        return self._children is None

    @property
    @abstractmethod
    def external_indices(self) -> tuple[Index, ...]:
        """Get the external indices (those that are not summed over)."""
        pass

    @property
    @abstractmethod
    def internal_indices(self) -> tuple[Index, ...]:
        """Get the internal indices (those that are summed over)."""
        pass

    @property
    def rank(self) -> int:
        """Get the rank."""
        return len(self.external_indices)

    @property
    @abstractmethod
    def disjoint(self) -> bool:
        """Return whether the object is disjoint."""
        pass

    @abstractmethod
    def copy(self, *args: Any, **kwargs: Any) -> Base:
        """Return a copy of the object with optionally updated attributes."""
        pass

    @abstractmethod
    def map_indices(self, mapping: dict[Index, Index]) -> Base:
        """Return a copy of the object with the indices mapped according to some dictionary.

        Args:
            mapping: map between old indices and new indices.

        Returns:
            Object with mapped indices.
        """
        pass

    def permute_indices(self, permutation: tuple[int, ...] | Permutation) -> Base:
        """Return a copy of the object with the indices permuted according to some permutation.

        Args:
            permutation: permutation to apply.

        Returns:
            Object with permuted indices.
        """
        if isinstance(permutation, tuple):
            perm = permutation
            sign = 1
        else:
            perm = permutation.permutation
            sign = permutation.sign
        indices = self.external_indices
        mapping = dict(zip(indices, (indices[i] for i in perm)))
        return sign * self.map_indices(mapping)

    @abstractmethod
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
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def expand(self) -> ExpandedAddLayer:
        """Expand the object into the minimally nested format.

        Output has the form Add[Mul[Tensor | Scalar]].

        Returns:
            Object in expanded format.
        """
        pass

    @abstractmethod
    def collect(self) -> Base:
        """Collect like terms in the top layer of the object.

        Returns:
            Object with like terms collected.
        """
        pass

    @abstractmethod
    def squeeze(self) -> Base:
        """Squeeze the object by removing any redundant algebraic operations.

        Returns:
            Object with redundant operations removed.
        """
        pass

    @abstractmethod
    def as_sympy(self) -> Any:
        """Return a sympy representation of the object.

        Returns:
            Object in sympy format.
        """
        pass

    @classmethod
    @abstractmethod
    def from_sympy(cls, data: Any) -> Any:
        """Return an object loaded from a sympy representation.

        Returns:
            Object loaded from sympy representation.
        """
        pass

    @classmethod
    def from_string(cls, string: str) -> Base:
        """Return an object loaded from a string representation.

        Returns:
            Object loaded from string representation.
        """
        from albert.misc import from_string

        return from_string(string)

    def __hash__(self) -> int:
        """Return the hash of the object.

        Returns:
            The integer hash of the object.

        Note:
            The fields of the hashable representation are defined by the `_hashable_fields` method.
            For subclasses of `Base`, the default fields are as follows:

                - Rank of the object (number of external indices).
                - External indices of the object.
                - Internal indices of the object.
                - Score of the interface class. This is used to distinguish between subclasses of
                    `Base`. The score is used to define the order of precedence in the interface
                    hierarchy (`Tensor` < `Scalar` < `Add` < `Mul`).
                - Name of the object.
                - Number of children.
                - Children of the object.

            If two objects are equal for each of these fields, they will have the same hash.
        """
        return super().__hash__()

    def _hashable_fields(self) -> Iterable[SerialisedField]:
        """Yield fields of the hashable representation."""
        yield self.rank
        yield self.external_indices
        yield self.internal_indices
        yield len(self._penalties)
        for penalty in self._penalties:
            yield penalty(self)
        yield self._score
        yield getattr(self, "name", "~")
        yield len(self._children) if self._children is not None else 0
        if self._children:
            yield from self._children

    @classmethod
    @abstractmethod
    def from_json(cls, data: Any) -> Base:
        """Return an object loaded from a JSON representation.

        Returns:
            Object loaded from JSON representation.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        pass

    def tree_repr(
        self,
        connectors: tuple[str, str, str, str] = ("─", "│", "├", "└"),
        indent: int = 4,
    ) -> str:
        """Return a string representation of the tree structure.

        Args:
            connectors: Characters to use for drawing the tree structure. The characters are, in
                order: (0) horizontal connection, (1) vertical connection, (2) vertical connection
                with horizontal branch, (3) vertical terminus with horizontal branch.
            indent: Number of spaces to use for indentation.

        Returns:
            String representation of the tree structure.
        """
        # Minimum indent of 2 to fit the connectors
        indent = max(indent, 2)

        def to_str(node: Base) -> str:
            if node._children:
                return node.__class__.__name__
            return str(node)

        def walk(node: Base, prefix: str = "", is_last: bool = True) -> None:
            nonlocal result  # type: ignore[misc]
            connector = connectors[3 if is_last else 2] + connectors[0] * (indent - 2) + " "
            result += f"{prefix}{connector}{to_str(node)}\n"
            if not node._children:
                return
            spacing = (connectors[1] if not is_last else " ") + " " * (indent - 1)
            next_prefix = f"{prefix}{spacing}"
            for i, child in enumerate(node._children):
                walk(child, next_prefix, i == len(node._children) - 1)

        # Build the string representation
        result = f"{to_str(self)}\n"
        for i, child in enumerate(self._children or []):
            walk(child, "", i == len(self._children or []) - 1)

        return result.rstrip()

    @abstractmethod
    def __add__(self, other: Base) -> Base:
        """Add two objects."""
        pass

    def __radd__(self, other: Base) -> Base:
        """Add two objects."""
        return self + other

    def __sub__(self, other: Base) -> Base:
        """Subtract two objects."""
        return self + (-1 * other)

    def __rsub__(self, other: Base) -> Base:
        """Subtract two objects."""
        return other + (-1 * self)

    @abstractmethod
    def __mul__(self, other: Base | float) -> Base:
        """Multiply two objects."""
        pass

    def __rmul__(self, other: Base | float) -> Base:
        """Multiply two objects."""
        return self * other  # Assume always commutative

    def __neg__(self) -> Base:
        """Negate the object."""
        return -1 * self
