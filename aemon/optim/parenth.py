"""Functions for parenthesising multiplications.
"""

import itertools

from aemon.tensor import Tensor
from aemon.algebra import Algebraic, Add, Mul
from aemon.optim.flops import count_flops


def partition(collection):
    """
    Partition a collection into all possible subsets of size 2 or
    greater.

    Ref: https://stackoverflow.com/questions/19368375

    Parameters
    ----------
    collection : iterable
        The collection to partition.

    Yields
    ------
    partition : list
        A partition of `collection`.
    """

    if len(collection) <= 2:
        yield [collection]
        return

    first = [collection[0], collection[1]]
    for smaller in partition(collection[2:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [first + subset] + smaller[n + 1 :]
        yield [first] + smaller


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
        if len(remain) == 1:
            yield path
            return

        for i, j in itertools.combinations(remain, 2):
            yield from _iterate(
                path + ((i, j),),
                remain - {j},
            )

    remain = set(range(n))
    path = tuple()
    yield from _iterate(path, remain)


def parenthesise(expr, path):
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
    args = list(expr.args)
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
def candidates_brute_force(expr, cost_fn=count_flops):
    """
    Find all possible parenthesising candidates for an algebraic
    expression.

    Parameters
    ----------
    expr : Mul or Tensor
        The expression to parenthesise.
    cost_fn : callable, optional
        The cost function to use. Defaults value is `count_flops`.

    Yields
    ------
    candidates : Mul
        The parenthesising candidates, sorted by their FLOP cost.
    """

    # If the expression is just a tensor, or only has one permutation,
    # there are no parenthesising candidates
    if isinstance(expr, Tensor) or len(expr.args) <= 2:
        yield expr
        return

    # Otherwise, find all possible parenthesising candidates
    candidates = []
    for path in parenthesisations(len(expr.args)):
        candidates.append(parenthesise(expr, path))

    # Sort the candidates by their cost
    candidates = sorted(candidates, key=cost_fn)

    # Yield the candidates
    yield from candidates
