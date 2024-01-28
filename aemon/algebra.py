"""Tensor algebra.
"""

import itertools
from functools import cached_property
from collections import defaultdict
from numbers import Number

from aemon import config
from aemon.base import Base


class Algebraic(Base):
    """Algebraic base class.
    """

    def __new__(cls, *args):
        """Create the object.
        """

        # Filter the arguments
        args = cls._filter_args(args)

        if len(args) == 0:
            # If there are no arguments, return zero
            return cls.default_value()
        elif len(args) == 1:
            # If there is one argument, return it instead
            return args[0]
        else:
            # Otherwise, create the object
            algebraic = super().__new__(cls)
            algebraic.args = args
            return algebraic

    @staticmethod
    def _filter_args(cls, args):
        """Filter the arguments.
        """
        return args

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
        args = []
        for arg in self.args:
            if not isinstance(arg, Number):
                arg = arg.map_indices(mapping)
            args.append(arg)
        return self.copy(*args)

    def hashable(self, coefficient=True):
        """Return a hashable representation of the object.
        """
        return (
            len(self.args),
            len(self.external_indices),
            self.__class__.__name__,
            self.coefficient if coefficient else None,
            tuple(
                arg.hashable() if isinstance(arg, Base) else arg
                for arg in self.without_coefficient().args
            ),
        )

    def canonicalise(self):
        """Canonicalise the object.
        """
        args = [arg.canonicalise() if isinstance(arg, Base) else arg for arg in self.args]
        expression = self.copy(*args)
        return expression


class Add(Algebraic):
    """Addition of tensors.
    """

    @classmethod
    def _filter_args(cls, args):
        """Filter the arguments.
        """

        # Check the external indices match
        indices = set(args[0].external_indices)
        for arg in args:
            if set(arg.external_indices) != indices:
                raise ValueError(
                    f"Incompatible external indices for {cls.__class__.__name__}: "
                    f"{arg.external_indices} != {indices}"
                )

        # Collect equivalent terms
        factors = defaultdict(list)
        for arg in args:
            factors[arg.without_coefficient()].append(arg.coefficient)
        args = [sum(f) * arg for arg, f in factors.items() if abs(sum(f)) > config.ZERO]

        return args

    @staticmethod
    def default_value():
        """Return the value of the object when it has no arguments.
        """
        return 0

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

    @property
    def coefficient(self):
        """Return the coefficient of the object.
        """
        return 1

    def without_coefficient(self):
        """Return the object without the coefficient.
        """
        return self

    def canonicalise(self):
        """Canonicalise the object.
        """
        args = [arg.canonicalise() if isinstance(arg, Base) else arg for arg in self.args]
        args = sorted(args)
        expression = self.copy(*args)
        return expression

    def expand(self):
        """Expand the parentheses.
        """

        # If the expression is just a tensor, remove the parentheses
        if len(self.args) == 1:
            return self.args[0].expand()

        # Recursively expand the parentheses
        args = []
        for arg in self.args:
            if isinstance(arg, Base):
                arg = arg.expand()
            if isinstance(arg, Add):
                args.extend(arg.args)
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

    def __add__(self, other):
        """Add two tensors.
        """
        if not isinstance(other, Algebraic):
            return Add(*self.args, other)
        elif isinstance(other, Add):
            return Add(*self.args, *other.args)
        else:
            return Add(self, other)

    def __radd__(self, other):
        """Add two tensors.
        """
        if not isinstance(other, Algebraic):
            return Add(other, *self.args)
        elif isinstance(other, Add):
            return Add(*other.args, *self.args)
        else:
            return Add(other, self)


class Mul(Algebraic):
    """Contraction of tensors or scalars.
    """

    @classmethod
    def _filter_args(cls, args):
        """Filter the arguments.
        """

        # Check we have proper Einstein notation
        counts = defaultdict(int)
        for arg in args:
            if isinstance(arg, Number):
                continue
            for index in arg.dummy_indices:
                counts[index] += 1
            for index in arg.external_indices:
                counts[index] += 1
        if any(count > 2 for count in counts.values()):
            raise ValueError(
                f"{cls.__class__.__name__} only supports Einstein notation, "
                "i.e. each index must appear at most twice."
            )

        # Collect factors
        factor = 1
        non_factors = []
        for arg in args:
            if isinstance(arg, Number):
                factor *= arg
            else:
                non_factors.append(arg)
        if abs(factor) <= config.ZERO:
            return [0]
        elif abs(factor - 1) <= config.ZERO:
            args = non_factors
        else:
            args = [factor] + non_factors

        return args

    @staticmethod
    def default_value():
        """Return the value of the object when it has no arguments.
        """
        return 1

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

    @property
    def coefficient(self):
        """Return the coefficient of the object.
        """
        for arg in self.args:
            if isinstance(arg, Number):
                return arg
        return 1

    def without_coefficient(self):
        """Return the object without the coefficient.
        """
        args = [arg for arg in self.args if not isinstance(arg, Number)]
        return self.copy(*args)

    def canonicalise(self):
        """Canonicalise the object.
        """
        args = [arg.canonicalise() if isinstance(arg, Base) else arg for arg in self.args]
        expression = self.copy(*args)
        return expression

    def expand(self):
        """Expand the parentheses.
        """

        # If the expression is just a tensor, remove the parentheses
        if len(self.args) == 1:
            return self.args[0].expand()

        # Recursively expand the parentheses
        args = []
        for arg in self.args:
            if isinstance(arg, Base):
                arg = arg.expand()
            if len(args) == 0:
                args.append(arg)
            elif isinstance(arg, Add):
                args = tuple(a * b for a, b in list(itertools.product(args, arg.args)))
            else:
                args = tuple(a * arg for a in args)

        return Add(*args)

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

    def __mul__(self, other):
        """Contract two tensors.
        """
        if not isinstance(other, Algebraic):
            return Mul(*self.args, other)
        elif isinstance(other, Mul):
            return Mul(*self.args, *other.args)
        else:
            return Mul(self, other)

    def __rmul__(self, other):
        """Contract two tensors.
        """
        if not isinstance(other, Algebraic):
            return Mul(other, *self.args)
        elif isinstance(other, Mul):
            return Mul(*other.args, *self.args)
        else:
            return Mul(other, self)
