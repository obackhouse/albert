"""Common subexpression elimination.
"""

import itertools
from collections import defaultdict

from einfun.tensor import Tensor
from einfun.optim.cost import count_flops, memory_cost
from einfun.optim.parenth import get_candidates


def _find_groups(*muls, n):
    """
    Find a list of n-fold groups of tensors involved in products, for a
    series of expressions.

    Scales as O(k m^n), where k is the number of expressions, m is the
    number of tensors in each expression, and n is the number of tensors
    in each group.

    Parameters
    ----------
    *muls : list of Mul
        A list of multiplications.
    n : int
        The number of tensors in each group. For example, if `n=2`, the
        function returns all pairs of tensors involved in products.

    Returns
    -------
    pairs : dict of tuple: set
        A dictionary mapping each n-fold group of tensors to a set of
        indices expressions in which they appear as a product.
    """

    # Initialise the dictionary
    groups = defaultdict(set)

    # Loop over the expressions
    for i, mul in enumerate(muls):
        tensor_indices = [i for i, arg in enumerate(mul.args) if isinstance(arg, Tensor)]
        for group in itertools.combinations(tensor_indices, n):
            groups[group].add(i)

    return groups


def cse(
    *exprs,
    method=None,
    cost_fn=count_flops,
    memory_fn=memory_cost,
    sizes=None,
    memory_limit=None,
):
    """
    Perform common subexpression elimination on a series of
    expressions.

    Parameters
    ----------
    *exprs : list of tuple of (Tensor, Algebraic)
        A series of expressions to be optimized. For each element, the
        first element of the tuple is the output tensor, and the second
        element is the algebraic expression.
    method : str, optional
        The method to use for finding parenthesising candidates. The
        available methods are {`"brute"`, `"greedy"`, `"branch"`}.
        Default value is `None`, which is equivalent to `"brute"` for
        inputs with 4 or fewer terms, and `"greedy"` for inputs with
        more than 4 terms.
    cost_fn : callable, optional
        The cost function to use. Defaults value is `count_flops`.
    memory_fn : callable, optional
        The memory function to use. Default value is `memory_cost`.
    sizes : dict, optional
        A dictionary mapping indices to their sizes. If not provided,
        all indices are assumed to have size `10`. Default value is
        `None`.
    memory_limit : int, optional
        The maximum memory usage allowed, in units of the native
        floating point size of the tensors. If not provided, the memory
        usage is not limited. Default value is `None`.

    Returns
    -------
    exprs : list of tuple of (Tensor, Algebraic)
        The optimized expressions.
    """

    pass
