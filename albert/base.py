"""Base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, cast

from albert.hashing import InternTable

if TYPE_CHECKING:
    from typing import Any, Iterable

    from typing_extensions import Self

    from albert.index import Index
    from albert.symmetry import Permutation
    from albert.types import EvaluatorArrayDict, SerialisedField

T = TypeVar("T", bound="Base")
TypeOrFilter = Optional[type[T] | tuple[type[T], ...] | Callable[["Base"], bool]]


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
        if self._hash is not None and other._hash is not None:
            if self._hash != other._hash:
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


class Traversal(Enum):
    """Enum for traversal."""

    PREORDER = 0
    """Pre-order traversal (node before children)."""

    POSTORDER = 1
    """Post-order traversal (children before node)."""


def _matches_filter(node: Base, type_filter: TypeOrFilter[Base]) -> bool:
    """Check if a node matches a type filter.

    Args:
        node: Node to check.
        type_filter: Type filter to check against.

    Returns:
        Whether the node matches the type filter.
    """
    if type_filter is None:
        return True
    if isinstance(type_filter, tuple) or isinstance(type_filter, type):
        return isinstance(node, type_filter)
    return type_filter(node)  # type: ignore


def _sign_penalty(base: Base) -> int:
    """Return a penalty for the sign in scalars in a base object.

    Args:
        base: Base object to check.

    Returns:
        Penalty for the sign.
    """
    if not base.children:
        return 0
    penalty = 1
    if base.children:
        for child in base.children:
            if hasattr(child, "value"):
                penalty *= 1 if getattr(child, "value") < 0 else -1
    return penalty


class Base(Serialisable):
    """Base class for algebraic types."""

    _score: int
    _children: Optional[tuple[Base, ...]]
    _penalties: tuple[Callable[[Base], int], ...] = (_sign_penalty,)
    _internal_indices: tuple[Index, ...]
    _external_indices: tuple[Index, ...]

    @classmethod
    @abstractmethod
    def factory(cls: type[Base], *args: Any, **kwargs: Any) -> Base:
        """Factory method to create a new object.

        Args:
            args: Positional arguments to pass to the constructor.
            kwargs: Keyword arguments to pass to the constructor.

        Returns:
            Algebraic object. In general, `factory` methods may return objects of a different type
            to the class they are called on.
        """
        pass

    @property
    def is_leaf(self) -> bool:
        """Get whether the object is a leaf in a tree."""
        return not bool(self.children)

    @property
    def children(self) -> tuple[Base, ...]:
        """Get the children of the node."""
        return self._children or ()

    def _search(
        self,
        level: int,
        type_filter: TypeOrFilter[T],
        order: Traversal,
    ) -> Iterable[tuple[T, int]]:
        """Depth-first search through the tree.

        Args:
            level: Current level in the tree.
            type_filter: Type to filter by.
            order: Order to traverse the tree.

        Yields:
            Elements of the `_children` tuple, recursively, along with their level in the tree.
        """
        if order == Traversal.PREORDER and _matches_filter(self, type_filter):
            yield cast(T, self), level
        if self.children is not None:
            for child in self.children:
                yield from child._search(level + 1, type_filter, order)
        if order == Traversal.POSTORDER and _matches_filter(self, type_filter):
            yield cast(T, self), level

    def search(
        self,
        type_filter: TypeOrFilter[T],
        order: Traversal = Traversal.PREORDER,
        depth: Optional[int] = None,
    ) -> Iterable[T]:
        """Depth-first search through the tree.

        Args:
            type_filter: Type to filter by.
            order: Order to traverse the tree.
            depth: Maximum (relative) depth to search to.

        Yields:
            Elements of the `_children` tuple, recursively.
        """
        for node, level in self._search(0, type_filter, order):
            if depth is not None and level > depth:
                continue
            yield node

    def find(
        self,
        type_filter: TypeOrFilter[T],
        order: Traversal = Traversal.PREORDER,
        depth: Optional[int] = None,
    ) -> Optional[T]:
        """Find the first node matching a type filter.

        Args:
            type_filter: Type to filter by.
            order: Order to traverse the tree.
            depth: Maximum (relative) depth to search to.

        Returns:
            First element of the `_children` tuple matching the type filter, recursively.
        """
        for node in self.search(type_filter=type_filter, order=order, depth=depth):
            return node
        return None

    def apply(
        self,
        function: Callable[[T], Base],
        type_filter: TypeOrFilter[T],
    ) -> Base:
        """Apply a function to nodes.

        Args:
            function: Functon to apply.
            type_filter: Type of node to apply to.

        Returns:
            Object after applying function (if applicable).
        """
        if self.children:
            children = tuple(child.apply(function, type_filter) for child in self.children)
            self = self.copy(*children)
        if _matches_filter(self, type_filter):
            return function(cast(T, self))
        return self

    @abstractmethod
    def delete(
        self,
        type_filter: TypeOrFilter[Base],
    ) -> Base:
        """Delete nodes (set its value to zero) matching a type filter.

        Args:
            type_filter: Type of node to delete.

        Returns:
            Object after deleting nodes (if applicable).
        """
        pass

    @abstractmethod
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
        pass

    @property
    def external_indices(self) -> tuple[Index, ...]:
        """Get the external indices (those that are not summed over)."""
        return self._external_indices

    @property
    def internal_indices(self) -> tuple[Index, ...]:
        """Get the internal indices (those that are summed over)."""
        return self._internal_indices

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
    def map_indices(self, mapping: dict[Index, Index]) -> Self:
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
    def expand(self) -> Base:
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
        yield getattr(self, "symmetry", None) is not None
        yield getattr(self, "symmetry", ())
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

    @abstractmethod
    def copy(self, *args: Any, **kwargs: Any) -> Base:
        """Return a copy of the object with optionally updated attributes."""
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
            if node.children:
                return node.__class__.__name__
            return str(node)

        def walk(node: Base, prefix: str = "", is_last: bool = True) -> None:
            nonlocal result  # type: ignore[misc]
            connector = connectors[3 if is_last else 2] + connectors[0] * (indent - 2) + " "
            result += f"{prefix}{connector}{to_str(node)}\n"
            if not node.children:
                return
            spacing = (connectors[1] if not is_last else " ") + " " * (indent - 1)
            next_prefix = f"{prefix}{spacing}"
            for i, child in enumerate(node.children):
                walk(child, next_prefix, i == len(node.children) - 1)

        # Build the string representation
        result = f"{to_str(self)}\n"
        for i, child in enumerate(self.children or []):
            walk(child, "", i == len(self.children or []) - 1)

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


_INTERN_TABLE = InternTable[Base]()
