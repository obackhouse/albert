"""Permutation symmetry."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, TypedDict

from albert.base import Serialisable

if TYPE_CHECKING:
    from typing import Iterable

    from albert.base import Base, SerialisedField


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


class Permutation(Serialisable):
    """Class for a permutation.

    Args:
        permutation: Permutation.
        sign: Sign of the permutation.
    """

    def __init__(self, permutation: tuple[int, ...], sign: int):
        """Initialise the permutation."""
        self._permutation = tuple(permutation)
        self._sign = sign
        self._hash = None

    @property
    def permutation(self) -> tuple[int, ...]:
        """Get the permutation."""
        return self._permutation

    @property
    def sign(self) -> int:
        """Get the sign."""
        return self._sign

    @property
    def rank(self) -> int:
        """Get the rank."""
        return len(self._permutation)

    def __call__(self, obj: Base) -> Base:
        """Apply the permutation to the object.

        Args:
            obj: Object to permute.

        Returns:
            Permuted object.
        """
        return obj.permute_indices(self._permutation) * self._sign

    def _hashable_fields(self) -> Iterable[SerialisedField]:
        """Yield fields of the hashable representation."""
        yield self.__class__.__name__
        yield self.rank
        yield from self._permutation
        yield self._sign

    def as_json(self) -> _PermutationJSON:
        """Return a JSON representation of the object.

        Returns:
            Object in JSON format.
        """
        return {
            "_type": self.__class__.__name__,
            "_module": self.__class__.__module__,
            "permutation": self._permutation,
            "sign": self._sign,
        }

    @classmethod
    def from_json(cls, data: _PermutationJSON) -> Permutation:
        """Return an object loaded from a JSON representation.

        Returns:
            Object loaded from JSON representation.
        """
        return cls(data["permutation"], data["sign"])

    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        return f"{self.__class__.__name__}({self._permutation}, {self._sign})"

    def __add__(self, other: Permutation) -> Permutation:
        """Append permutations."""
        perm = self._permutation + tuple(p + len(self._permutation) for p in other._permutation)
        sign = self._sign * other._sign
        return Permutation(perm, sign)

    def __mul__(self, other: Permutation) -> Permutation:
        """Compose permutations."""
        perm = tuple(self._permutation[p] for p in other._permutation)
        sign = self._sign * other._sign
        return Permutation(perm, sign)


class Symmetry(Serialisable):
    """Permutation symmetry group.

    Args:
        permutations: Permutations.
    """

    def __init__(self, *permutations: Permutation):
        """Initialise the symmetry."""
        self._permutations = permutations

    @property
    def permutations(self) -> tuple[Permutation, ...]:
        """Get the permutations."""
        return self._permutations

    def __call__(self, obj: Base) -> Iterable[Base]:
        """Iterate over the permutations of the object.

        Args:
            obj: Object to permute.

        Yields:
            Permuted objects.
        """
        for permutation in self._permutations:
            yield permutation(obj)

    def _hashable_fields(self) -> Iterable[SerialisedField]:
        """Yield fields of the hashable representation."""
        yield self.__class__.__name__
        yield len(self._permutations)
        yield from self._permutations

    def as_json(self) -> _SymmetryJSON:
        """Return a JSON representation of the object.

        Returns:
            Object in JSON format.
        """
        return {
            "_type": self.__class__.__name__,
            "_module": self.__class__.__module__,
            "permutations": tuple(p.as_json() for p in self._permutations),
        }

    @classmethod
    def from_json(cls, data: _SymmetryJSON) -> Symmetry:
        """Return an object loaded from a JSON representation.

        Returns:
            Object loaded from JSON representation.
        """
        return cls(*[Permutation.from_json(p) for p in data["permutations"]])

    def __repr__(self) -> str:
        """Return a string representation.

        Returns:
            String representation.
        """
        return f"{self.__class__.__name__}({', '.join(map(str, self._permutations))})"

    def __add__(self, other: Symmetry) -> Symmetry:
        """Append permutations."""
        return Symmetry(*(self._permutations + other._permutations))

    def __mul__(self, other: Symmetry) -> Symmetry:
        """Compose permutations."""
        return Symmetry(*(p * q for p, q in zip(self._permutations, other._permutations)))


def symmetric_group(*permutations: tuple[int, ...]) -> Symmetry:
    """Generate the symmetric group from permutations.

    Args:
        permutations: Permutations.

    Returns:
        Symmetry group.
    """
    return Symmetry(*[Permutation(perm, 1) for perm in permutations])


def non_symmetric_group(n: int) -> Symmetry:
    """Generate the non-symmetric group of `n` objects.

    Args:
        n: Number of objects.

    Returns:
        Symmetry group.
    """
    return Symmetry(Permutation(tuple(range(n)), 1))


def fully_symmetric_group(n: int) -> Symmetry:
    """Generate the fully symmetric group of `n` objects.

    Args:
        n: Number of objects.

    Returns:
        Symmetry group.
    """
    return Symmetry(*[Permutation(perm, 1) for perm in itertools.permutations(range(n))])


def fully_antisymmetric_group(n: int) -> Symmetry:
    """Generate the fully antisymmetric group of `n` objects.

    Args:
        n: Number of objects.

    Returns:
        Symmetry group.
    """

    def _permutations(seq: list[int]) -> list[list[int]]:
        """Generate permutations of a sequence."""
        if not seq:
            return [[]]

        items = []
        for i, item in enumerate(_permutations(seq[:-1])):
            if i % 2:
                inds = range(len(item) + 1)
            else:
                inds = range(len(item), -1, -1)
            items += [item[:i] + seq[-1:] + item[i:] for i in inds]

        return items

    permutations = [
        Permutation(tuple(item), -1 if i % 2 else 1)
        for i, item in enumerate(_permutations(list(range(n))))
    ]

    return Symmetry(*permutations)
