"""Functions to determine FLOP costs for algebraic expressions.
"""

from collections import defaultdict

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
