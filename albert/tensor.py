"""Base class for tensors.
"""

from numbers import Number

from albert import check_dependency, sympy
from albert.algebra import Add, Mul
from albert.base import Base
from albert.symmetry import Permutation


def defer_to_algebra(method):
    """
    Decorate overloaded operators to return `NotImplemented` if the
    other operand is not a tensor or scalar. This allows the
    overloaded operation to be deferred to the algebraic classes.
    """

    def wrapper(self, other):
        if not isinstance(other, (Number, Tensor)):
            return NotImplemented
        return method(self, other)

    return wrapper


class Tensor(Base):
    """Base class for tensors."""

    def __init__(self, *indices, name=None, symmetry=None):
        """Initialise the object."""
        self.indices = indices
        self.name = name
        self.symmetry = symmetry
        self._symbol = None

    def __repr__(self):
        """Return the representation of the object."""
        name = self.name if self.name else self.__class__.__name__
        indices = ",".join([str(x) for x in self.indices])
        return f"{name}[{indices}]"

    @property
    def rank(self):
        """Return the rank of the object."""
        return len(self.indices)

    @property
    def external_indices(self):
        """Return the external indices of the object."""
        return self.indices

    @property
    def dummy_indices(self):
        """Return the dummy indices of the object."""
        return tuple()

    @property
    def coefficient(self):
        """Return the coefficient of the object."""
        return 1

    @property
    def disjoint(self):
        """Return whether the object is disjoint."""
        return False

    def without_coefficient(self):
        """Return the object without the coefficient."""
        return self

    def copy(self, *indices, name=None, symmetry=None):
        """Copy the object."""
        if not indices:
            indices = self.indices
        if not name:
            name = self.name
        if not symmetry:
            symmetry = self.symmetry
        tensor = self.__class__(*indices, name=name, symmetry=symmetry)
        tensor._symbol = self._symbol
        return tensor

    def map_indices(self, mapping):
        """Map the indices of the object."""
        indices = [mapping.get(index, index) for index in self.indices]
        return self.copy(*indices)

    def permute_indices(self, permutation):
        """Permute the indices of the object."""
        if isinstance(permutation, Permutation):
            factor = permutation.sign
            permutation = permutation.permutation
        else:
            factor = 1
        indices = [self.indices[i] for i in permutation]
        return self.copy(*indices) * factor

    def hashable(self):
        """Return a hashable representation of the object."""
        return (
            0,  # penalty
            1,  # size
            self.rank,
            self.__class__.__name__,
            (
                (self.name,),
                self.external_indices,
                self.dummy_indices,
                self.symmetry.hashable() if self.symmetry else None,
            ),
        )

    def canonicalise(self):
        """Canonicalise the object."""
        if not self.symmetry:
            return self
        return min(self.symmetry(self))

    def expand(self):
        """Expand the object."""
        return self

    def nested_view(self):
        """
        Return a view of the expression, with the parentheses expanded,
        as a nested list. The first layer is the addition, the second
        layer is the multiplication, and the third layer are the tensors
        and scalars.

        This is for compatibility with algebraic expressions, and always
        simply returns `[[self]]`.

        Returns
        -------
        nested : list of list of Tensor
            The nested view of the expression.
        """
        return [[self]]

    def as_symbol(self):
        """Return a symbol for the object."""
        if self._symbol is None:
            return Symbol(self.name, symmetry=self.symmetry)
        else:
            return self._symbol

    @check_dependency("sympy")
    def as_sympy(self):
        """Return a sympy representation of the object."""

        indices = tuple(sympy.Symbol(str(index)) for index in self.indices)
        symbol = self.as_symbol().as_sympy()

        return symbol[indices]

    @property
    def args(self):
        """Return the arguments of the object."""
        return (self,)

    @defer_to_algebra
    def __add__(self, other):
        """Add two tensors."""
        return Add(self, other)

    @defer_to_algebra
    def __radd__(self, other):
        """Add two tensors."""
        return Add(other, self)

    @defer_to_algebra
    def __sub__(self, other):
        """Subtract two tensors."""
        return Add(self, -1 * other)

    @defer_to_algebra
    def __rsub__(self, other):
        """Subtract two tensors."""
        return Add(other, -1 * self)

    @defer_to_algebra
    def __mul__(self, other):
        """Multiply two tensors."""
        return Mul(self, other)

    @defer_to_algebra
    def __rmul__(self, other):
        """Multiply two tensors."""
        return Mul(other, self)

    def __neg__(self):
        """Negate the tensor."""
        return -1 * self


class Symbol:
    """Constructor for tensors."""

    DESIRED_RANK = None
    Tensor = Tensor

    def __init__(self, name, symmetry=None):
        """Initialise the object."""
        self.name = name
        self.symmetry = symmetry

    def __getitem__(self, indices):
        """Return a tensor."""
        if isinstance(indices, str) and len(indices) == self.DESIRED_RANK:
            indices = tuple(indices)
        elif not isinstance(indices, tuple):
            indices = (indices,)
        if self.DESIRED_RANK is not None:
            if len(indices) != self.DESIRED_RANK:
                raise ValueError(
                    f"{self.__class__.__name__} expected {self.DESIRED_RANK} indices, "
                    f"got {len(indices)}."
                )
        return self.Tensor(*indices, name=self.name, symmetry=self.symmetry)

    def hashable(self):
        """Return a hashable representation of the object."""
        return (self.name, self.symmetry.hashable() if self.symmetry else None)

    @check_dependency("sympy")
    def as_sympy(self):
        """Return a sympy representation of the object."""

        return sympy.IndexedBase(self.name)

    def __hash__(self):
        """Return a hash of the object."""
        return hash(self.hashable())

    def __repr__(self):
        """Return the representation of the object."""
        return self.name
