"""Parenthesisation of tensor products."""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING

from albert import _default_sizes
from albert.tensor import Tensor
from albert.algebra import Mul

if TYPE_CHECKING:
    from typing import Any, Optional, Generator

    import cotengra

    from albert.index import Index

    T = tuple[int, ...]


def _format_mul(expr: Mul) -> Mul:
    """Check the ``Mul`` expression is valid and return it."""
    expanded = expr.expand()
    if len(expanded._children) != 1:
        raise ValueError("Expression {expr} is not a valid tensor product.")
    return expanded._children[0]


def _get_inputs_and_output(
    expr: Mul,
    sizes: dict[str | None, float],
) -> tuple[tuple[T, ...], T, dict[int, float]]:
    """Get the einsum inputs and output from a tensor product.

    Args:
        expr: The product to analyse.
        sizes: The sizes of the spaces in the expression.

    Returns:
        A tuple of the einsum inputs, output, and sizes.
    """
    expr = _format_mul(expr)
    memo: dict[Index, int] = {}
    inputs = []
    for tensor in expr.search_leaves(Tensor):
        inds = []
        for ind in tensor.indices:
            if ind not in memo:
                memo[ind] = len(memo)
            inds.append(memo[ind])
        inputs.append(tuple(inds))

    output = tuple(sorted(memo[ind] for ind in expr.external_indices))
    sizes = {memo[ind]: sizes[ind.space] for ind in memo}

    return tuple(inputs), output, sizes


def find_optimal_path(
    expr: Mul,
    sizes: Optional[dict[str | None, int]] = None,
) -> cotengra.ContractionTree:
    """Find the optimal path for a tensor product.

    Args:
        expr: The product to parenthesise.
        sizes: The sizes of the spaces in the expression.

    Returns:
        The optimal contraction tree.
    """
    expr = _format_mul(expr)
    if sizes is None:
        sizes = _default_sizes

    import cotengra

    # Expand the product
    if not isinstance(expr, Mul):
        raise TypeError("Expression must be a Mul.")
    expr, = expr.expand()._children

    # Optimise the contraction path
    inputs, output, sizes = _get_inputs_and_output(expr, sizes)
    opt = cotengra.OptimalOptimizer()
    tree = opt.search(inputs, output, sizes)

    return tree


def generate_paths_exhaustive(
    expr: Mul, sizes: Optional[dict[str | None, int]] = None
) -> Generator[cotengra.ContractionTree, None, None]:
    """Generate all possible paths for a tensor product.

    Args:
        expr: The product to parenthesise.
        sizes: The sizes of the spaces in the expression.

    Yields:
        All possible contraction trees.
    """
    expr = _format_mul(expr)
    num_tensors = len(list(expr.search_leaves(Tensor)))
    if num_tensors < 2:
        return
    if num_tensors > 5:
        warnings.warn(
            "Generating all possible paths scales extremely poorly. Use with caution.",
            RuntimeWarning,
            sacklevel=2,
        )
    if sizes is None:
        sizes = _default_sizes

    import cotengra

    def _recurse(path: list[T], n: int) -> None:
        if n == 1:
            yield tuple(path)
            return
        for i, j in itertools.combinations(range(n), 2):
            path.append((i, j))
            yield from _recurse(path, n - 1)
            path.pop()

    # Get all paths
    inputs, output, sizes = _get_inputs_and_output(expr, sizes)
    for path in _recurse([], num_tensors):
        tree = cotengra.ContractionTree.from_path(inputs, output, sizes, path=path)
        yield tree


def generate_paths_approximate(
    expr: Mul,
    sizes: Optional[dict[str | None, float]] = None,
    max_samples: int = 8,
    **opt_kwargs: Any,
) -> Generator[cotengra.ContractionTree, None, None]:
    """Generate approximate paths for a tensor product.

    Args:
        expr: The product to parenthesise.
        sizes: The sizes of the spaces in the expression.
        max_samples: The maximum number of paths to generate.
        **opt_kwargs: Additional keyword arguments to pass to the optimizers.

    Yields:
        Approximate contraction trees.
    """
    expr = _format_mul(expr)
    num_tensors = len(list(expr.search_leaves(Tensor)))
    if num_tensors < 2:
        return
    if sizes is None:
        sizes = _default_sizes

    import cotengra

    # Get trial paths
    inputs, output, sizes = _get_inputs_and_output(expr, sizes)
    opt = cotengra.HyperOptimizer(max_repeats=max_samples, **opt_kwargs)
    trial_fn, trial_args = opt.setup(inputs, output, sizes)
    repeats_start = opt._repeats_start + len(opt.scores)
    repeats = range(repeats_start, repeats_start + max_samples)
    trials = (opt._gen_results_parallel if opt._pool is not None else opt._gen_results)(
        repeats, trial_fn, trial_args
    )

    # Yield unique trees
    seen: set[tuple[T, ...]] = set()
    for trial in trials:
        if len(seen) >= max_samples:
            break
        tree: cotengra.ContractionTree = trial["tree"]
        if tree.get_path() not in seen:
            seen.add(tree.get_path())
            yield tree

    # Clean up
    opt._maybe_cancel_futures()
    del opt


def generate_paths(
    expr: Mul,
    sizes: Optional[dict[str | None, int]] = None,
    max_samples: int = 8,
    **opt_kwargs: Any,
) -> Generator[cotengra.ContractionTree, None, None]:
    """Generate paths for a tensor product.

    Args:
        expr: The product to parenthesise.
        sizes: The sizes of the spaces in the expression.
        max_samples: The maximum number of paths to generate if not exhaustive.
        **opt_kwargs: Additional keyword arguments to pass to the optimizers.

    Yields:
        Contraction trees.
    """
    expr = _format_mul(expr)
    num_tensors = len(list(expr.search_leaves(Tensor)))
    if num_tensors < 40:
        yield find_optimal_path(expr, sizes=sizes)
        max_samples -= 1
    if num_tensors < 6 and max_samples > 0:
        trees = list(generate_paths_exhaustive(expr, sizes=sizes))
        costs = [tree.total_cost(log=None) for tree in trees]
        for _, tree in sorted(zip(costs, trees), key=lambda x: x[0])[:max_samples]:
            yield tree
    elif max_samples > 0:
        yield from generate_paths_approximate(
            expr, sizes=sizes, max_samples=max_samples, **opt_kwargs
        )
