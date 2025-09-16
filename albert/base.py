"""Base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Hashable, Protocol

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Optional, TypeVar

    from albert.algebra import ExpandedAddLayer
    from albert.index import Index
    from albert.symmetry import Permutation, Symmetry

    T = TypeVar("T", bound="Base")


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


class Base(Serialisable):
    """Base class for algebraic types."""

    _interface: type[IBase]
    _children: Optional[tuple[Base, ...]]

    @staticmethod
    def _spin_penalty(indices: tuple[Index, ...]) -> int:
        """Return a penalty for the configuration of spins in a set of indices.

        Args:
            indices: Indices to check.

        Returns:
            Penalty for the configuration of spins.
        """
        spins = tuple(index.spin if index.spin else "" for index in indices)
        penalty = 0
        for i in range(len(spins) - 1):
            penalty += int(spins[i] == spins[i + 1]) * 2
        if spins and spins[0] != min(spins):
            penalty += 1
        return penalty

    @staticmethod
    def _sign_penalty(children: tuple[Base, ...]) -> int:
        """Return a penalty for the sign in scalars.

        Args:
            children: Children to check.

        Returns:
            Penalty for the sign.
        """
        penalty = 1
        for child in children:
            if isinstance(child, IScalar):
                penalty *= 1 if child._value < 0 else -1
        return -penalty

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
                    `Base`, however subclasses of i.e. `Tensor` will all have the same interface
                    class (i.e. `ITensor`) unless `_interface` is overridden. The score is used to
                    define the order of precedence in the interface hierarchy (`ITensor` < `IScalar`
                    < `IAdd` < `IMul`).
                - Name of the object.
                - Number of children.
                - Children of the object.

            If two objects are equal for each of these fields, they will have the same hash.
        """
        return super().__hash__()

    def _hashable_fields(self) -> Iterable[SerialisedField]:
        """Yield fields of the hashable representation."""
        yield self.rank
        yield self._spin_penalty(self.external_indices)
        yield self.external_indices
        yield self.internal_indices
        yield self._sign_penalty(self._children) if self._children else 0
        yield _SCORES[self._interface]
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
    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of expressions resulting from the conversion.
        """
        pass

    @abstractmethod
    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Expression resulting from the conversion.
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


class IBase(Base):
    """Interface class.

    Note:
        This class is entirely abstract and is only used to indicate the type of the object within
        the ``Base`` class itself.
    """

    pass


class IScalar(IBase):
    """Interface class for scalar types.

    Args:
        value: Value of the scalar.

    Note:
        This class is entirely abstract and is only used to indicate the type of the object within
        the ``Base`` class itself.
    """

    _value: float


class ITensor(IBase):
    """Interface class for a tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.

    Note:
        This class is entirely abstract and is only used to indicate the type of the object within
        the ``Base`` class itself.
    """

    _indices: tuple[Index, ...]
    _name: str
    _symmetry: Optional[Symmetry]


class IAlgebraic(IBase):
    """Interface class for an algebraic operations.

    Args:
        children: Children to the operation.

    Note:
        This class is entirely abstract and is only used to indicate the type of the object within
        the ``Base`` class itself.
    """

    pass


class IAdd(IAlgebraic):
    """Interface class for addition.

    Note:
        This class is entirely abstract and is only used to indicate the type of the object within
        the ``Base`` class itself.
    """

    pass


class IMul(IAlgebraic):
    """Interface class for multiplication.

    Note:
        This class is entirely abstract and is only used to indicate the type of the object within
        the ``Base`` class itself.
    """

    pass


# Precedence scores for ordering
_SCORES = {
    ITensor: 0,
    IScalar: 1,
    IAdd: 2,
    IMul: 3,
}
_SCORES_REVERSE = {v: k for k, v in _SCORES.items()}
