"""Tensor algebra.
"""

import itertools
from functools import cached_property
from collections import defaultdict
from numbers import Number

from aemon.base import Base


class Algebraic(Base):
    """Algebraic base class.
    """

    def __init__(self, *args):
        """Initialise the object.
        """
        self.args = args

    def __add__(self, other):
        """Add two tensors.
        """
        return Add(self, other)

    def __radd__(self, other):
        """Add two tensors.
        """
        return Add(other, self)

    def __mul__(self, other):
        """Multiply two tensors.
        """
        return Mul(self, other)

    def __rmul__(self, other):
        """Multiply two tensors.
        """
        return Mul(other, self)

    def copy(self, *args):
        """Copy the object.
        """
        if not args:
            args = self.args
        return self.__class__(*args)

    def map_indices(self, mapping):
        """Map the indices of the object.
        """
        args = [arg.map_indices(mapping) for arg in self.args]
        return self.copy(*args)

    def hashable(self):
        """Return a hashable representation of the object.
        """
        return tuple(arg.hashable() if isinstance(arg, Base) else arg for arg in self.args)

    def canonicalise(self):
        """Canonicalise the object.
        """
        args = [arg.canonicalise() if isinstance(arg, Base) else arg for arg in self.args]
        expression = self.copy(*args)
        return expression


class Add(Algebraic):
    """Addition of tensors.
    """

    def __init__(cls, *args):
        """Create a new object.
        """
        indices = args[0].external_indices
        for arg in args:
            if arg.external_indices != indices:
                raise ValueError(
                    f"Incompatible external indices for {cls.__class__.__name__}: "
                    f"{arg.external_indices} != {indices}"
                )
        return Algebraic.__init__(cls, *args)

    @property
    def external_indices(self):
        """Return the external indices of the object.
        """
        return self.args[0].external_indices

    @property
    def dummy_indices(self):
        """Return the dummy indices of the object.
        """
        return tuple()

    def canonicalise(self):
        """Canonicalise the object.
        """
        args = [arg.canonicalise() if isinstance(arg, Base) else arg for arg in self.args]
        args = sorted(args, key=lambda arg: arg.hashable())
        expression = self.copy(*args)
        return expression

    def expand(self):
        """Expand the parentheses.
        """
        args = []
        for arg in self.args:
            if isinstance(arg, Base):
                args.append(arg.expand())
            else:
                args.append(arg)
        return self.copy(*args)

    def __repr__(self):
        """Return the representation of the object.
        """
        atoms = []
        for arg in self.args:
            if isinstance(arg, Algebraic):
                atoms.append(f"({repr(arg)})")
            else:
                atoms.append(repr(arg))
        string = " + ".join(atoms)
        return string


class Mul(Add):
    """Multiplication of tensors.

    Not to be confused with contraction, this class handles the indexing
    of tensor symbols that are multiplied together.
    """

    def expand(self):
        """Expand the parentheses.
        """
        mul_args = [arg.expand() if isinstance(arg, Base) else arg for arg in self.args]
        add_args = [arg.args if isinstance(arg, Add) else [arg] for arg in mul_args]
        args = [Mul(*arg) for arg in itertools.product(*add_args)]
        return Add(*args) if len(args) > 1 else args[0]

    def __repr__(self):
        """Return the representation of the object.
        """
        atoms = []
        for arg in self.args:
            if isinstance(arg, Algebraic):
                atoms.append(f"({repr(arg)})")
            else:
                atoms.append(repr(arg))
        string = " * ".join(atoms)
        return string


class Dot(Algebraic):
    """Contraction of tensors or scalars.
    """

    def __init__(cls, *args):
        """Create a new object.
        """
        return Algebraic.__init__(cls, *args)

    @cached_property
    def external_indices(self):
        """Return the external indices of the object.
        """
        counts = defaultdict(int)
        for arg in self.args:
            if isinstance(arg, Number):
                continue
            for index in arg.external_indices:
                counts[index] += 1
        return tuple(index for index, count in counts.items() if count == 1)

    @cached_property
    def dummy_indices(self):
        """Return the dummy indices of the object.
        """
        counts = defaultdict(int)
        for arg in self.args:
            if isinstance(arg, Number):
                continue
            for index in arg.external_indices:
                counts[index] += 1
        return tuple(index for index, count in counts.items() if count > 1)

    def canonicalise(self):
        """Canonicalise the object.
        """
        args = [arg.canonicalise() if isinstance(arg, Base) else arg for arg in self.args]
        expression = self.copy(*args)
        return expression

    def expand(self):
        """Expand the parentheses.
        """
        dot_args = [arg.expand() if isinstance(arg, Base) else arg for arg in self.args]
        add_args = [arg.args if isinstance(arg, Add) else [arg] for arg in dot_args]
        args = [Dot(*arg) for arg in itertools.product(*add_args)]
        return Add(*args) if len(args) > 1 else args[0]

    def __repr__(self):
        """Return the representation of the object.
        """
        atoms = []
        for arg in self.args:
            if isinstance(arg, Algebraic):
                atoms.append(f"({repr(arg)})")
            else:
                atoms.append(repr(arg))
        string = " * ".join(atoms)
        return string
