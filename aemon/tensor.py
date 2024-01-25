"""Base class for tensors.
"""

from numbers import Number

from aemon.base import Base
from aemon.algebra import Add, Mul, Dot


def defer_to_algebra(method):
    """
    Decorator for overloaded operators, to return `NotImplemented` if
    the other operand is not a tensor or scalar. This allows the
    overloaded operation to be deferred to the algebraic classes.
    """
    def wrapper(self, other):
        if not isinstance(other, (Number, Tensor)):
            return NotImplemented
        return method(self, other)
    return wrapper


class Tensor(Base):
    """Base class for tensors.
    """

    def __init__(self, *indices, name=None, symmetry=None):
        """Initialise the object.
        """
        self.indices = indices
        self.name = name
        self.symmetry = symmetry

    def __repr__(self):
        """Return the representation of the object.
        """
        name = self.name if self.name else self.__class__.__name__
        indices = ", ".join([repr(x) for x in self.indices])
        return f"{name}[{indices}]"

    @property
    def rank(self):
        """Return the rank of the object.
        """
        return len(self.indices)

    @property
    def external_indices(self):
        """Return the external indices of the object.
        """
        return self.indices

    @property
    def dummy_indices(self):
        """Return the dummy indices of the object.
        """
        return tuple()

    @property
    def coefficient(self):
        """Return the coefficient of the object.
        """
        return 1

    def without_coefficient(self):
        """Return the object without the coefficient.
        """
        return self

    def copy(self, *indices, name=None, symmetry=None):
        """Copy the object.
        """
        if not indices:
            indices = self.indices
        if not name:
            name = self.name
        if not symmetry:
            symmetry = self.symmetry
        return Tensor(*indices, name=name, symmetry=symmetry)

    def map_indices(self, mapping):
        """Map the indices of the object.
        """
        indices = [mapping.get(index, index) for index in self.indices]
        return self.copy(*indices)

    def hashable(self):
        """Return a hashable representation of the object.
        """
        return (
            self.name,
            self.external_indices,
            self.dummy_indices,
            self.symmetry.hashable() if self.symmetry else None,
        )

    def canonicalise(self):
        """Canonicalise the object.
        """
        if not self.symmetry:
            return self
        return min(self.symmetry(self))

    def expand(self):
        """Expand the object.
        """
        return self

    @defer_to_algebra
    def __add__(self, other):
        """Add two tensors.
        """
        return Add(self, other)

    @defer_to_algebra
    def __radd__(self, other):
        """Add two tensors.
        """
        return Add(other, self)

    @defer_to_algebra
    def __mul__(self, other):
        """Multiply two tensors.
        """
        return Mul(self, other)

    @defer_to_algebra
    def __rmul__(self, other):
        """Multiply two tensors.
        """
        return Mul(other, self)

    @defer_to_algebra
    def __matmul__(self, other):
        """Contract two tensors.
        """
        return Dot(self, other)

    @defer_to_algebra
    def __rmatmul__(self, other):
        """Contract two tensors.
        """
        return Dot(other, self)
