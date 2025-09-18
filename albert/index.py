"""Indices for tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from albert.base import Serialisable, SerialisedField

if TYPE_CHECKING:
    from typing import Any, Iterable, Optional


def _to_greek(name: str) -> str:
    """Convert a spin channel name to a greek letter."""
    return chr(0x3B1 + ord(name) - ord("a"))


class _IndexJSON(TypedDict):
    """Type for JSON representation of an index."""

    _type: str
    _module: str
    name: str
    spin: Optional[str]
    space: Optional[str]


def from_list(
    names: list[str],
    spins: list[Optional[str]] | Optional[str] = None,
    spaces: list[Optional[str]] | Optional[str] = None,
) -> list[Index]:
    """Construct a list of indices from lists of names, spins, and spaces.

    Args:
        names: List of names.
        spins: List of spins. Can be a single value for all indices.
        spaces: List of spaces. Can be a single value for all indices.

    Returns:
        List of indices.
    """
    if spins is None:
        spins = [None] * len(names)
    elif isinstance(spins, str):
        spins = [spins] * len(names)
    if spaces is None:
        spaces = [None] * len(names)
    elif isinstance(spaces, str):
        spaces = [spaces] * len(names)
    return [Index(name, spin=spin, space=space) for name, spin, space in zip(names, spins, spaces)]


class Index(Serialisable):
    """Class for indices.

    Args:
        name: The name of the index.
        spin: The spin of the index.
        space: The space of the index.
    """

    __slots__ = ("_name", "_spin", "_space", "_hash")

    _name: str
    _spin: Optional[str]
    _space: Optional[str]

    def __init__(self, name: str, spin: Optional[str] = None, space: Optional[str] = None):
        """Initialise the object."""
        self._name = name
        self._spin = spin
        self._space = space
        self._hash = None

    @property
    def name(self) -> str:
        """Get the name of the index."""
        return self._name

    @property
    def spin(self) -> Optional[str]:
        """Get the spin of the index."""
        return self._spin

    @property
    def space(self) -> Optional[str]:
        """Get the space of the index."""
        return self._space

    @property
    def category(self) -> tuple[str, str]:
        """Get the category of the index, a compound of space and spin."""
        space = self._space if self._space is not None else ""
        spin = self._spin if self._spin is not None else ""
        return (space, spin)

    def copy(
        self, name: Optional[str] = None, spin: Optional[str] = None, space: Optional[str] = None
    ) -> Index:
        """Return a copy of the object with some properties changed."""
        if name is None:
            name = self._name
        if spin is None:
            spin = self._spin
        if space is None:
            space = self._space
        return Index(name, spin=spin, space=space)

    def spin_flip(self) -> Index:
        """Return a copy of the object with the spin flipped."""
        spin = self.spin
        if spin in ("a", "b"):
            return self.copy(spin="a" if spin == "b" else "b")
        return self

    def as_sympy(self) -> Any:
        """Return a sympy representation of the object.

        Returns:
            Object in sympy format.
        """
        import sympy

        name = self._name
        if self._spin is not None:
            name += f"{self._spin}"
        if self._space is not None:
            name += f"{self._space}"

        return sympy.Symbol(name)

    @classmethod
    def from_sympy(cls, data: Any) -> Index:
        """Return an object loaded from a sympy representation.

        Returns:
            Object loaded from sympy representation.
        """
        name: str = data.name
        spin: Optional[str] = None
        space: Optional[str] = None
        if "%" in name:
            name, space = name.split("%")
        if "$" in name:
            name, spin = name.split("$")
        return cls(name, spin=spin, space=space)

    def as_json(self) -> _IndexJSON:
        """Return a JSON representation of the object.

        Returns:
            Object in JSON format.
        """
        return {
            "_type": self.__class__.__name__,
            "_module": self.__class__.__module__,
            "name": self._name,
            "spin": self._spin,
            "space": self._space,
        }

    @classmethod
    def from_json(cls, data: _IndexJSON) -> Index:
        """Return an object loaded from a JSON representation.

        Returns:
            Object loaded from JSON representation.
        """
        return cls(data["name"], spin=data["spin"], space=data["space"])

    def _hashable_fields(self) -> Iterable[SerialisedField]:
        """Yield fields of the hashable representation."""
        yield self.__class__.__name__
        yield self._space.lower() if self._space else ""
        yield self._space.isupper() if self._space else False
        yield self._spin if self._spin is not None else ""
        yield self._name

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        if self._spin in ("a", "b"):
            return f"{self._name}{_to_greek(self._spin)}"
        return self._name
