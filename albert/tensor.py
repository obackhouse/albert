"""Classes for tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from albert.algebra import Add, Mul
from albert.base import _INTERN_TABLE, Base, _matches_filter
from albert.index import Index
from albert.scalar import Scalar

if TYPE_CHECKING:
    from typing import Any, Optional

    from albert.base import TypeOrFilter
    from albert.symmetry import Permutation, Symmetry
    from albert.types import _TensorJSON

T = TypeVar("T", bound=Base)


class Tensor(Base):
    """Class for a tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
    """

    __slots__ = (
        "_indices",
        "_name",
        "_symmetry",
        "_hash",
        "_children",
        "_internal_indices",
        "_external_indices",
    )

    _score = 0

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        self._indices = indices
        self._name = name or self.__class__.__name__
        self._symmetry = symmetry
        self._hash = None
        self._children = None

        # Precompute indices
        self._external_indices = tuple(index for index in indices if indices.count(index) == 1)
        self._internal_indices = tuple(index for index in indices if indices.count(index) > 1)

    @classmethod
    def factory(
        cls: type[Tensor],
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ) -> Tensor:
        """Factory method to create a new object.

        Args:
            indices: Indices of the tensor.
            name: Name of the tensor.
            symmetry: Symmetry of the tensor.

        Returns:
            Algebraic object. In general, `factory` methods may return objects of a different type
            to the class they are called on.
        """
        if not issubclass(cls, Tensor):
            raise TypeError(f"cls must be a subclass of Tensor, got {cls}")

        # Build a key for interning
        key = (cls, indices, name, symmetry)

        def create() -> Tensor:
            return cls(*indices, name=name, symmetry=symmetry)

        return cast(Tensor, _INTERN_TABLE.get(key, create))

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
        return Scalar.factory(0.0) if _matches_filter(self, type_filter) else self

    @property
    def indices(self) -> tuple[Index, ...]:
        """Get the indices of the object."""
        return self._indices

    @property
    def name(self) -> str:
        """Get the name of the object."""
        return self._name

    @property
    def symmetry(self) -> Optional[Symmetry]:
        """Get the symmetry of the object."""
        return self._symmetry

    @property
    def disjoint(self) -> bool:
        """Return whether the object is disjoint."""
        return False

    def copy(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ) -> Tensor:
        """Return a copy of the object with optionally updated attributes.

        Args:
            indices: New indices.
            name: New name.
            symmetry: New symmetry.

        Returns:
            Copy of the object.
        """
        if not indices:
            indices = self.indices
        if name is None:
            name = self.name
        if symmetry is None:
            symmetry = self.symmetry
        return self.__class__(*indices, name=name, symmetry=symmetry)

    def map_indices(self, mapping: dict[Index, Index]) -> Tensor:
        """Return a copy of the object with the indices mapped according to some dictionary.

        Args:
            mapping: map between old indices and new indices.

        Returns:
            Object with mapped indices.
        """
        indices = tuple(mapping.get(index, index) for index in self.indices)
        return self.copy(*indices)

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
        indices = tuple(self.indices[i] for i in perm)
        return sign * self.copy(*indices)

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
        if self.symmetry is not None:
            best = min(self.symmetry(self))
        else:
            best = self

        if indices:
            from albert.canon import canonicalise_indices

            best = canonicalise_indices(best)

        return best

    def expand(self) -> Base:
        """Expand the object into the minimally nested format.

        Output has the form Add[Mul[Tensor | Scalar]].

        Returns:
            Object in expanded format.
        """
        return Add(Mul(self))

    def collect(self) -> Tensor:
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

        name = self.name
        indices = [index.as_sympy() for index in self.indices]

        if len(indices) == 0:
            return sympy.Symbol(name)
        else:
            return sympy.IndexedBase(name)[tuple(indices)]

    @classmethod
    def from_sympy(cls, data: Any) -> Tensor:
        """Return an object loaded from a sympy representation.

        Returns:
            Object loaded from sympy representation.
        """
        name: str = data.base.label
        indices = [Index.from_sympy(index) for index in data.indices]
        return cls(*indices, name=name)

    def as_json(self) -> _TensorJSON:
        """Return a JSON representation of the object.

        Returns:
            Object in JSON format.
        """
        return {
            "_type": self.__class__.__name__,
            "_module": self.__class__.__module__,
            "indices": tuple(index.as_json() for index in self.indices),
            "name": self.name,
            "symmetry": self.symmetry.as_json() if self.symmetry else None,
        }

    @classmethod
    def from_json(cls, data: _TensorJSON) -> Tensor:
        """Return an object loaded from a JSON representation.

        Returns:
            Object loaded from JSON representation.
        """
        indices = [Index.from_json(index) for index in data["indices"]]
        symmetry = None
        if data["symmetry"]:
            symmetry = Symmetry.from_json(data["symmetry"])
        return cls(*indices, name=data["name"], symmetry=symmetry)

    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        if len(self._indices) == 0:
            return self.name
        return f"{self.name}({','.join(map(str, self.indices))})"

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
            return Mul.factory(self, *other.children)
        return Mul.factory(self, other)
