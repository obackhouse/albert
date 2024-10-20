"""Tensor algebra.
"""

import itertools
import random
from collections import defaultdict
from functools import cached_property
from numbers import Number

import networkx as nx

from albert import check_dependency, config, sympy
from albert.base import Base
from albert.tree import Tree


class Algebraic(Base):
    """Algebraic base class."""

    def __new__(cls, *args):
        """Create the object."""

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
            algebraic._hashable = {}
            return algebraic

    @staticmethod
    def _filter_args(cls, args):
        """Filter the arguments."""
        return args

    def __add__(self, other):
        """Add two tensors."""
        return Add(self, other)

    def __radd__(self, other):
        """Add two tensors."""
        return Add(other, self)

    def __sub__(self, other):
        """Subtract two tensors."""
        return Add(self, -1 * other)

    def __rsub__(self, other):
        """Subtract two tensors."""
        return Add(other, -1 * self)

    def __mul__(self, other):
        """Multiply two tensors."""
        return Mul(self, other)

    def __rmul__(self, other):
        """Multiply two tensors."""
        return Mul(other, self)

    def __neg__(self):
        """Negate the tensor."""
        return -1 * self

    def copy(self, *args):
        """Copy the object."""
        if not args:
            args = self.args
        return self.__class__(*args)

    def map_indices(self, mapping):
        """Map the indices of the object."""
        args = []
        for arg in self.args:
            if not isinstance(arg, Number):
                arg = arg.map_indices(mapping)
            args.append(arg)
        return self.copy(*args)

    def permute_indices(self, permutation):
        """Permute the indices of the object."""
        args = []
        for arg in self.args:
            if not isinstance(arg, Number):
                arg = arg.permute_indices(permutation)
            args.append(arg)
        return self.copy(*args)

    def map_tensors(self, mapping):
        """Map the tensors of the object."""
        args = []
        for arg in self.args:
            if not isinstance(arg, Number):
                arg = arg.map_tensors(mapping)
            else:
                arg = mapping.get(arg, arg)
            args.append(arg)
        return self.copy(*args)

    def apply(self, function):
        """Apply a function to the object."""
        args = []
        for arg in self.args:
            if not isinstance(arg, Number):
                arg = arg.apply(function)
            args.append(arg)
        return self.copy(*args)

    def hashable(self, coefficient=True, penalty_function=None):
        """Return a hashable representation of the object."""

        # Return the cached hashable representation if available
        if (coefficient, penalty_function) in self._hashable:
            return self._hashable[(coefficient, penalty_function)]

        # Recursively get the hashable representation
        args = list(self.without_coefficient().args)
        hashable = tuple(
            arg.hashable(penalty_function=penalty_function) if isinstance(arg, Base) else arg
            for arg in args
        )

        # Add the algebraic information
        hashable += (self.coefficient if coefficient else None,)
        hashable += (len(self.external_indices),)
        hashable += (self.__class__.__name__,)

        # Flatten the tuple
        hashable = tuple(
            itertools.chain.from_iterable(
                (item if isinstance(item, tuple) else (item,)) for item in hashable
            )
        )

        # Cache the hashable representation
        self._hashable[(coefficient, penalty_function)] = hashable

        return hashable

    def canonicalise(self):
        """Canonicalise the object."""
        args = [arg.canonicalise() if isinstance(arg, Base) else arg for arg in self.args]
        expression = self.copy(*args)
        return expression

    def nested_view(self):
        """
        Return a view of the expression, with the parentheses expanded,
        as a nested list. The first layer is the addition, the second
        layer is the multiplication, and the third layer are the tensors
        and scalars.

        Returns
        -------
        nested : list of list of (Tensor or Number)
            The nested view of the expression.
        """

        # Expand the parentheses
        expr = self.expand()

        # Dissolve the addition layer
        if isinstance(expr, Add):
            nested = [[arg] for arg in expr.args]
        else:
            nested = [[expr]]

        # Dissolve the multiplication layer
        for i, arg in enumerate(nested):
            if isinstance(arg[0], Mul):
                nested[i] = arg[0].args

        return nested

    def as_tree(self, tree=None, seed=0):
        """
        Return a tree representation of the expression. The tree has a
        structure such as

            +
           / \
          w   *
             / \
            +   v
           /|\
          x y z

        for the expression `w + v * (x + y + z)`.

        Parameters
        ----------
        tree : Tree, optional
            The tree to add the expression to. If not provided, a new
            tree is created.
        seed : int, optional
            Seed to use for discerning identical tensors and factors in
            the tree. Mainly used in the internal recursive call.
            Default value is 0.

        Returns
        -------
        tree : Tree
            The tree representation of the expression.
        """

        # Initialise the tree
        if tree is None:
            tree = Tree()

        # Get the operator
        node = (self, seed)

        # Get the children
        seeds = [random.randint(0, 2**32) for _ in self.args]
        children = [(arg, seed) for arg, seed in zip(self.args, seeds)]

        # Add the operator node
        tree.add(
            node,
            name=self.__class__.__name__,
            children=children,
        )

        # Add the children
        for arg, child, seed in zip(self.args, children, seeds):
            if isinstance(arg, Algebraic):
                arg.as_tree(tree=tree, seed=seed)
            elif isinstance(arg, Number):
                tree.add(child, name=repr(arg))
            else:
                tree.add(child, name=arg.name)

        return tree

    def as_json(self):
        """Return a JSON serialisable representation of the object."""
        return {
            "_type": self.__class__.__name__,
            "_path": self.__module__,
            "args": [arg.as_json() if isinstance(arg, Base) else arg for arg in self.args],
        }

    @classmethod
    def from_json(cls, data):
        """Return an object from a JSON serialisable representation.

        Notes
        -----
        This method is non-recursive and the dictionary members should
        already be parsed.
        """
        return cls(*data["args"])

    def replace(self, expr_old, expr_new):
        """
        Replace a subexpression.

        Parameters
        ----------
        expr_old : Algebraic
            The subexpression to replace.
        expr_new : Algebraic
            The new subexpression.
        """

        # Check the subexpression is exactly the expression
        if expr_old == self:
            return expr_new

        # Replace the subexpression
        args = []
        for arg in self.args:
            if arg == expr_old:
                args.append(expr_new)
            elif isinstance(arg, Algebraic):
                arg.replace(expr_old, expr_new)
                args.append(arg)
            else:
                args.append(arg)

        return self.copy(*args)


class Add(Algebraic):
    """Addition of tensors."""

    @classmethod
    def _filter_args(cls, args):
        """Filter the arguments."""

        # Remove zero terms
        args = [arg for arg in args if not isinstance(arg, Number) or abs(arg) > config.ZERO]

        # Check the external indices match
        indices = set(args[0].external_indices) if len(args) else set()
        for arg in args:
            if not isinstance(arg, Number):
                if set(arg.external_indices) != indices:
                    raise ValueError(
                        f"Incompatible external indices for {cls.__class__.__name__}: "
                        f"{arg.external_indices} != {indices}"
                    )

        # Collect equivalent terms
        factors = defaultdict(list)
        for arg in args:
            if not isinstance(arg, Number):
                factors[arg.without_coefficient()].append(arg.coefficient)
            else:
                factors[arg].append(1)
        args = [sum(f) * arg for arg, f in factors.items() if abs(sum(f)) > config.ZERO]

        return args

    @staticmethod
    def default_value():
        """Return the value of the object when it has no arguments."""
        return 0

    @property
    def external_indices(self):
        """Return the external indices of the object."""
        return self.args[0].external_indices

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
        """Return whether the arguments are disjoint."""
        return False

    def without_coefficient(self):
        """Return the object without the coefficient."""
        return self

    def canonicalise(self):
        """Canonicalise the object."""
        args = [arg.canonicalise() if isinstance(arg, Base) else arg for arg in self.args]
        args = sorted(args)
        expression = self.copy(*args)
        return expression

    def expand(self):
        """Expand the parentheses."""

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

    @check_dependency("sympy")
    def as_sympy(self):
        """Return a `sympy` representation of the object."""

        # Convert the arguments to `sympy` objects
        args = [arg.as_sympy() if isinstance(arg, Base) else arg for arg in self.args]

        # Return the sum
        return sympy.Add(*args)

    def __repr__(self):
        """Return the representation of the object."""
        atoms = []
        for arg in self.args:
            if isinstance(arg, Algebraic):
                atoms.append(f"({repr(arg)})")
            else:
                atoms.append(repr(arg))
        string = " + ".join(atoms)
        return string

    def __add__(self, other):
        """Add two tensors."""
        if not isinstance(other, Algebraic):
            return Add(*self.args, other)
        elif isinstance(other, Add):
            return Add(*self.args, *other.args)
        else:
            return Add(self, other)

    def __radd__(self, other):
        """Add two tensors."""
        if not isinstance(other, Algebraic):
            return Add(other, *self.args)
        elif isinstance(other, Add):
            return Add(*other.args, *self.args)
        else:
            return Add(other, self)


class Mul(Algebraic):
    """Contraction of tensors or scalars."""

    @classmethod
    def _filter_args(cls, args):
        """Filter the arguments."""

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
        """Return the value of the object when it has no arguments."""
        return 1

    @cached_property
    def external_indices(self):
        """Return the external indices of the object."""
        counts = defaultdict(int)
        for arg in self.args:
            if isinstance(arg, Number):
                continue
            for index in arg.external_indices:
                counts[index] += 1
        return tuple(index for index, count in counts.items() if count == 1)

    @cached_property
    def dummy_indices(self):
        """Return the dummy indices of the object."""
        counts = defaultdict(int)
        for arg in self.args:
            if isinstance(arg, Number):
                continue
            for index in arg.external_indices:
                counts[index] += 1
        return tuple(index for index, count in counts.items() if count > 1)

    @property
    def coefficient(self):
        """Return the coefficient of the object."""
        for arg in self.args:
            if isinstance(arg, Number):
                return arg
        return 1

    @property
    def disjoint(self):
        """Return whether the arguments are disjoint."""
        return len(self.dummy_indices) == 0

    def without_coefficient(self):
        """Return the object without the coefficient."""
        args = [arg for arg in self.args if not isinstance(arg, Number)]
        return self.copy(*args)

    def canonicalise(self):
        """Canonicalise the object."""
        args = [arg.canonicalise() if isinstance(arg, Base) else arg for arg in self.args]
        expression = self.copy(*args)
        return expression

    def expand(self):
        """Expand the parentheses."""

        # If the expression is just a tensor, remove the parentheses
        if len(self.args) == 1:
            return self.args[0].expand()

        # Recursively expand the parentheses
        args = None
        for arg in self.args:
            if isinstance(arg, Base):
                arg = arg.expand()
            if args is None:
                if isinstance(arg, Add):
                    args = arg.args
                else:
                    args = [arg]
            elif isinstance(arg, Add):
                args = [a * b for a, b in list(itertools.product(args, arg.args))]
            else:
                args = [a * arg for a in args]

        return Add(*args)

    def as_graph(self):
        """
        Generate a `networkx` graph representation of the contractions
        between tensors in the expression.

        Returns
        -------
        graph : networkx.MultiGraph
            The graph representation of the expression.
        """

        # Expand any nested Mul objects
        args = self.args
        i = 0
        while i < len(args):
            if isinstance(args[i], Mul):
                args = args[:i] + args[i].args + args[i + 1 :]
            else:
                i += 1
        expanded = self.copy(*args)

        # Create the graph
        graph = nx.MultiGraph(hashable=lambda x: hash(x.hashable()))

        # Add the nodes
        for i, arg in enumerate(expanded.args):
            if not isinstance(arg, Number):
                graph.add_node(arg.hashable(), data=arg)

        # Add the edges
        for index in expanded.dummy_indices:
            # Find the tensors with the index -- FIXME improve
            tensors = []
            for i, arg in enumerate(expanded.args):
                if not isinstance(arg, Number) and index in arg.external_indices:
                    tensors.append(i)
            assert len(tensors) == 2

            # Find the position of the index in each tensor
            positions = []
            for i in tensors:
                positions.append(expanded.args[i].external_indices.index(index))

            # Add the edges
            graph.add_edge(
                expanded.args[tensors[0]].hashable(),
                expanded.args[tensors[1]].hashable(),
                data={
                    expanded.args[tensors[0]].hashable(): positions[0],
                    expanded.args[tensors[1]].hashable(): positions[1],
                },
            )

        return graph

    @check_dependency("sympy")
    def as_sympy(self):
        """Return a `sympy` representation of the object."""

        # Convert the arguments to `sympy` objects
        args = [arg.as_sympy() if isinstance(arg, Base) else arg for arg in self.args]

        # Return the product
        return sympy.Mul(*args)

    def __repr__(self):
        """Return the representation of the object."""
        atoms = []
        for arg in self.args:
            if isinstance(arg, Algebraic):
                atoms.append(f"({repr(arg)})")
            else:
                atoms.append(repr(arg))
        string = " * ".join(atoms)
        return string

    def __mul__(self, other):
        """Contract two tensors."""
        if not isinstance(other, Algebraic):
            return Mul(*self.args, other)
        elif isinstance(other, Mul):
            return Mul(*self.args, *other.args)
        else:
            return Mul(self, other)

    def __rmul__(self, other):
        """Contract two tensors."""
        if not isinstance(other, Algebraic):
            return Mul(other, *self.args)
        elif isinstance(other, Mul):
            return Mul(*other.args, *self.args)
        else:
            return Mul(other, self)
