"""Functions for parenthesising multiplications.
"""

import heapq
import itertools
import functools
from collections import defaultdict

from aemon.tensor import Tensor
from aemon.algebra import Algebraic, Add, Mul
from aemon.optim.cost import count_flops, memory_cost


def parenthesisations(n):
    """Iterate over all possible parenthesisations of `n` objects.

    Parameters
    ----------
    n : int
        The number of objects to parenthesise.

    Yields
    ------
    parenthesisation : tuple of tuple of int
        A tuple of tuples, where each tuple is a pair of indices that
        are contracted together. After the contraction of two indices,
        the result is considered to have the index of the first index
        in the pair.
    """

    def _iterate(path, remain):
        # If there is only one index left, we're done
        if len(remain) == 1:
            yield path
            return

        # Iterate over all possible pairs of the remaining indices,
        # remove the second index from the pair from the remaining
        # indices, and recurse
        for i, j in itertools.combinations(remain, 2):
            yield from _iterate(
                path + ((i, j),),
                remain - {j},
            )

    # Trigger the recursive iteration
    remain = set(range(n))
    path = tuple()
    yield from _iterate(path, remain)


def parenthesisations_greedy(n, cost):
    """
    Iterate over possible parenthesisations of `n` objects with a
    greedy algorithm.

    Parameters
    ----------
    n : int
        The number of objects to parenthesise.
    cost : callable
        A function that takes a parenthesisation and returns a cost.

    Yields
    ------
    parenthesisation : tuple of tuple of int
        A tuple of tuples, where each tuple is a pair of indices that
        are contracted together. After the contraction of two indices,
        the result is considered to have the index of the first index
        in the pair.
    """

    remain = set(range(n))
    path = tuple()
    while len(remain) > 1:
        min_cost = None
        best_pair = None

        # Iterate over all possible pairs of the remaining indices
        # and find the pair that minimises the cost
        for i, j in itertools.combinations(remain, 2):
            new_path = path + ((i, j),)
            new_cost = cost(new_path)

            if min_cost is None or new_cost < min_cost:
                min_cost = new_cost
                best_pair = (i, j)

        # Add the best pair to the path and remove the second index
        # from the pair from the remaining indices
        path += (best_pair,)
        remain -= {best_pair[1]}

    yield path


def parenthesisations_branch(n, cost, depth=4):
    """
    Iterate over parenthesisations of `n` objects with a branching
    algorithm.

    Parameters
    ----------
    n : int
        The number of objects to parenthesise.
    cost : callable
        A function that takes a parenthesisation and returns a cost.
    term_cost : callable
        A function that takes a pair of indices and returns the cost of
        contracting them.
    depth : int, optional
        The maximum depth to search. Default value is `3`.

    Yields
    ------
    parenthesisation : tuple of tuple of int
        A tuple of tuples, where each tuple is a pair of indices that
        are contracted together. After the contraction of two indices,
        the result is considered to have the index of the first index
        in the pair.
    """

    best_cost = (float("inf"), float("inf"))

    def _iterate(path, remain):
        nonlocal best_cost

        # If there is only one index left, we're done
        if len(remain) == 1:
            new_cost = cost(path)
            if new_cost < best_cost:
                best_cost = new_cost
            yield path
            return

        # Iterate over all possible pairs of the remaining indices
        candidates = []
        for i, j in itertools.combinations(remain, 2):
            new_path = path + ((i, j),)
            new_cost = cost(new_path, partial=True)

            # If the partial cost is already worse than the best cost,
            # we can skip this pair
            if new_cost > best_cost:
                continue

            heapq.heappush(candidates, (new_cost, (i, j)))

        # Iterate over the best `depth` candidates and recurse
        for b in range(depth):
            if not candidates:
                break
            _, (i, j) = heapq.heappop(candidates)
            yield from _iterate(
                path + ((i, j),),
                remain - {j},
            )

    # Trigger the recursive iteration
    remain = set(range(n))
    path = tuple()
    yield from _iterate(path, remain)


def parenthesise(expr, path, partial=False):
    """
    Parenthesise an algebraic expression.

    Parameters
    ----------
    expr : Mul or Tensor
        The expression to parenthesise.
    path : tuple of tuple of int
        A tuple of tuples, where each tuple is a pair of indices that
        are contracted together. After the contraction of two indices,
        the result is considered to have the index of the first index
        in the pair.

    Returns
    -------
    parenthesised : Mul
        The parenthesised expression.
    """
    if not partial:
        args = new_args = list(expr.args)
    else:
        args = list(expr.args)
        new_args = [None] * len(args)
    for i, j in path:
        args[i] = Mul(args[i], args[j])
        args[j] = None
    args = [arg for arg in args if arg is not None]
    return Mul(*args)


def check_input(expr):
    """
    Check that the input is either a single tensor or a `Mul` with no
    `Add` children.

    Parameters
    ----------
    expr : Mul or Tensor
        The expression to check.

    Raises
    ------
    TypeError
        If `expr` is not a `Mul`, or if it contains an `Add`.
    """

    if isinstance(expr, Algebraic) and any(isinstance(arg, Add) for arg in expr.args):
        raise TypeError(
            f"Algebraic expression must not contain any `Add`s for parenthesising. Use "
            "intermediate tensors in place of sums to avoid this."
        )


def _check_input(func):
    """Decorate a function to check its input.
    """

    def wrapper(expr, *args, **kwargs):
        check_input(expr)
        return func(expr, *args, **kwargs)

    return wrapper


@_check_input
def get_candidates(
    expr,
    method=None,
    cost_fn=count_flops,
    memory_fn=memory_cost,
    sizes=None,
    prefer_memory=False,
    return_costs=False,
):
    """
    Find all possible parenthesising candidates for an algebraic
    expression.

    Parameters
    ----------
    expr : Mul or Tensor
        The expression to parenthesise.
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
    prefer_memory : bool, optional
        Whether to prefer parenthesising candidates with lower memory
        overhead as priority over lower FLOP cost. Default value is
        `False`.
    return_costs : bool, optional
        Whether to return the costs of the parenthesising candidates
        along with the candidates themselves. Default value is `False`.

    Yields
    ------
    candidates : Mul
        The parenthesising candidates, sorted by their FLOP cost.
    """

    # If the expression is just a tensor, or only has one permutation,
    # there are no parenthesising candidates
    if isinstance(expr, Tensor) or len(expr.args) <= 2:
        if return_costs:
            yield expr, (cost_fn(expr, sizes=sizes), memory_fn(expr, sizes=sizes))
        else:
            yield expr
        return

    # Get the cost function
    if prefer_memory:
        expr_cost = lambda expr: (memory_fn(expr, sizes=sizes), cost_fn(expr, sizes=sizes))
    else:
        expr_cost = lambda expr: (cost_fn(expr, sizes=sizes), memory_fn(expr, sizes=sizes))
    cost = lambda path, partial=False: expr_cost(parenthesise(expr, path, partial=partial))
    #term_cost = lambda i, j: expr_cost(Mul(expr.args[i], expr.args[j]))

    # Get the parenthesising function
    if method is None:
        method = "greedy" if len(expr.args) > 4 else "brute"
    if method == "brute":
        func = parenthesisations
        cost_args = ()
    elif method == "greedy":
        func = parenthesisations_greedy
        cost_args = (cost,)
    elif method == "branch":
        func = parenthesisations_branch
        cost_args = (cost,)
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Find parenthesising candidates
    candidates = []
    for path in func(len(expr.args), *cost_args):
        candidates.append(parenthesise(expr, path))

    # Sort the candidates by their cost
    costs = [expr_cost(candidate) for candidate in candidates]
    candidates, costs = zip(*sorted(zip(candidates, costs), key=lambda x: x[1]))

    # Yield the candidates
    if return_costs:
        yield from zip(candidates, costs)
    else:
        yield from candidates
