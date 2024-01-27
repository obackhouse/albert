"""Functions to determine FLOP costs for algebraic expressions.
"""

from collections import defaultdict
from numbers import Number

from aemon.algebra import Algebraic, Add, Mul


def count_flops(expr, sizes=None, mul=1, add=1):
    """
    Simply count the number of FLOPs in an expression.

    Parameters
    ----------
    expr : Algebraic
        The expression to count FLOPs in.
    sizes : dict, optional
        A dictionary mapping indices to their sizes. If not provided,
        all indices are assumed to have size `10`. Default value is
        `None`.
    mul : int, optional
        The relative cost of a multiplication. Default value is `1`.
    add : int, optional
        The relative cost of an addition. Default value is `1`.

    Returns
    -------
    flops : int
        The number of FLOPs in the expression.
    """

    # Get the index sizes
    if sizes is None:
        sizes = defaultdict(lambda: 10)

    # Count the FLOPs in the expression
    flops = 1
    for index in expr.external_indices:
        flops *= sizes[index]
    for index in expr.dummy_indices:
        flops *= sizes[index]

    # Factor by the operation costs
    if isinstance(expr, Add) and add != 1:
        flops *= add * (len(expr.args) - 1)
    elif isinstance(expr, Mul) and mul != 1:
        flops *= mul * (len(expr.args) - 1)

    # Count the FLOPs recursively
    for arg in expr.args:
        if isinstance(arg, Algebraic):
            flops += count_flops(arg, sizes=sizes, mul=mul, add=add)

    return flops


def memory_cost(expr, sizes=None):
    """
    Simply count the amount of memory required to store an expression.

    Parameters
    ----------
    expr : Algebraic
        The expression to count memory usage in.
    sizes : dict, optional
        A dictionary mapping indices to their sizes. If not provided,
        all indices are assumed to have size `10`. Default value is
        `None`.

    Returns
    -------
    memory : int
        The amount of memory required to execute the expression.
    """

    # Get the index sizes
    if sizes is None:
        sizes = defaultdict(lambda: 10)

    # Count the memory usage in the result
    memory_out = 1
    for index in expr.external_indices:
        memory_out *= sizes[index]

    # Count the memory usage in the arguments
    memory_in = 0
    for arg in expr.args:
        if not isinstance(arg, Number):
            memory_in_arg = 1
            for index in arg.external_indices:
                memory_in_arg *= sizes[index]
            memory_in += memory_in_arg
        else:
            memory_in += 1

    # Find the maximum memory usage recursively
    memory = memory_in + memory_out
    for arg in expr.args:
        if isinstance(arg, Algebraic):
            memory = max(memory, memory_cost(arg, sizes=sizes))

    return memory
