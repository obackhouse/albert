"""Classes for tensors."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, TypedDict, TypeVar, cast

from albert.algebra import Add, ExpandedAddLayer, ExpandedMulLayer, Mul, _compose_add, _compose_mul
from albert.base import Base, ITensor
from albert.canon import canonicalise_indices
from albert.index import Index
from albert.scalar import Scalar

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

    from albert.index import _IndexJSON
    from albert.symmetry import Symmetry, _SymmetryJSON

T = TypeVar("T", bound=Base)


class _TensorJSON(TypedDict):
    """Type for JSON representation of a tensor."""

    _type: str
    _module: str
    indices: tuple[_IndexJSON, ...]
    name: str
    symmetry: Optional[_SymmetryJSON]


class Tensor(Base):
    """Class for a tensor.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
    """

    _interface = ITensor

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

    @cached_property
    def external_indices(self) -> tuple[Index, ...]:
        """Get the external indices (those that are not summed over)."""
        return tuple(index for index in self._indices if self._indices.count(index) == 1)

    @cached_property
    def internal_indices(self) -> tuple[Index, ...]:
        """Get the internal indices (those that are summed over)."""
        return tuple(index for index in self._indices if self._indices.count(index) > 1)

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
            indices = self._indices
        if name is None:
            name = self._name
        if symmetry is None:
            symmetry = self._symmetry
        return self.__class__(*indices, name=name, symmetry=symmetry)

    def map_indices(self, mapping: dict[Index, Index]) -> Tensor:
        """Return a copy of the object with the indices mapped according to some dictionary.

        Args:
            mapping: map between old indices and new indices.

        Returns:
            Object with mapped indices.
        """
        indices = tuple(mapping.get(index, index) for index in self._indices)
        return self.copy(*indices)

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
        if isinstance(self, node_type):
            return function(cast(T, self))
        return self

    def canonicalise(self, indices: bool = False) -> Base:
        """Canonicalise the object.

        The results of this function for equivalent representations should be equal.

        Args:
            indices: Whether to canonicalise the indices of the object. When `True`, this is
                performed for the outermost call in recursive calls.

        Returns:
            Object in canonical format.
        """
        if self._symmetry is not None:
            best = min(self._symmetry(self))
        else:
            best = self
        if indices:
            best = canonicalise_indices(best)
        return best

    def expand(self) -> ExpandedAddLayer:
        """Expand the object into the minimally nested format.

        Output has the form Add[Mul[Tensor | Scalar]].

        Returns:
            Object in expanded format.
        """
        mul = cast(ExpandedMulLayer, Mul(self))
        add = cast(ExpandedAddLayer, Add(mul))
        return add

    def collect(self) -> Tensor:
        """Collect like terms in the top layer of the object.

        Returns:
            Object with like terms collected.
        """
        return self

    def as_sympy(self) -> Any:
        """Return a sympy representation of the object.

        Returns:
            Object in sympy format.
        """
        import sympy

        name = self._name
        indices = [index.as_sympy() for index in self._indices]

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
            "indices": tuple(index.as_json() for index in self._indices),
            "name": self._name,
            "symmetry": self._symmetry.as_json() if self._symmetry else None,
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

    def as_uhf(self, target_rhf: bool = False) -> tuple[Base, ...]:
        """Convert the indices without spin to indices with spin.

        Indices that start without spin are assumed to be spin orbitals.

        Args:
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

        Returns:
            Tuple of expressions resulting from the conversion.
        """
        raise NotImplementedError(
            "Conversion methods `as_rhf` and `as_uhf` are implemented for the subclasses of "
            "`Tensor` in `albert.qc` modules."
        )

    def as_rhf(self) -> Base:
        """Convert the indices with spin to indices without spin.

        Indices that are returned without spin are spatial orbitals.

        Returns:
            Expression resulting from the conversion.
        """
        raise NotImplementedError(
            "Conversion methods `as_rhf` and `as_uhf` are implemented for the subclasses of "
            "`Tensor` in `albert.qc` modules."
        )

    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        if len(self._indices) == 0:
            return self._name
        return f"{self._name}({','.join(map(str, self._indices))})"

    def __add__(self, other: Base | float) -> Base:
        """Add two objects."""
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if isinstance(other, Add):
            return _compose_add(self, *other._children)
        return _compose_add(self, other)

    def __mul__(self, other: Base | float) -> Base:
        """Multiply two objects."""
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if isinstance(other, Mul):
            return _compose_mul(self, *other._children)
        return _compose_mul(self, other)
