"""Parenthesisation of tensor products."""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING, cast

import cotengra
import opt_einsum

from albert import _default_sizes
from albert.algebra import Mul
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Generator, Optional

    from albert.index import Index

    T = tuple[int, ...]


def _format_mul(expr: Mul) -> Mul:
    """Check the ``Mul`` expression is valid and return it."""
    expanded = expr.expand()
    if len(expanded.children) != 1:
        raise ValueError("Expression {expr} is not a valid tensor product.")
    return cast(Mul, expanded.children[0])


def _get_inputs_and_output(
    expr: Mul,
    sizes: dict[str | None, int],
) -> tuple[tuple[T, ...], T, dict[int, int]]:
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
    for tensor in expr.search(Tensor):
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
    max_cpu_scaling: dict[tuple[str, ...], int] | None = None,
    max_ram_scaling: dict[tuple[str, ...], int] | None = None,
) -> cotengra.ContractionTree:
    """Find the optimal path for a tensor product.

    Args:
        expr: The product to parenthesise.
        sizes: The sizes of the spaces in the expression.
        max_cpu_scaling: The maximum CPU scaling for the spaces in the expression.
        max_ram_scaling: The maximum RAM scaling for the spaces in the expression.

    Returns:
        The optimal contraction tree.
    """
    expr = _format_mul(expr)
    if sizes is None:
        sizes = _default_sizes

    # Expand the product
    if not isinstance(expr, Mul):
        raise TypeError("Expression must be a Mul.")
    (expr,) = cast(tuple[Mul], expr.expand().children)

    # Get constrained cost function
    sizes_dummy = dict(zip(sizes.keys(), itertools.count(1, 1)))
    dummy_map = {v: k for k, v in sizes_dummy.items()}
    sizes_map = {sizes_dummy[k]: sizes[k] for k in sizes_dummy}

    def cost(
        cost1: int,
        cost2: int,
        i1_union_i2: set[int],
        size_dict: list[int],
        cost_cap: int,
        s1: int,
        s2: int,
        xn: dict[int, Any],
        g: int,
        all_tensors: int,
        inputs: list[set[int]],
        i1_cut_i2_wo_output: set[int],
        memory_limit: Optional[int],
        contract1: int | tuple[int],
        contract2: int | tuple[int],
    ) -> None:
        # Get the cost scaling
        scaling: dict[str | None, int] = {}
        for i in i1_union_i2:
            c = dummy_map[size_dict[i]]
            scaling[c] = scaling.get(c, 0) + 1

        # Check the cost scaling
        if max_cpu_scaling is not None:
            for cs, n in max_cpu_scaling.items():
                if sum(scaling.get(c, 0) for c in cs) > n:
                    return

        # Get the real cost
        size_dict_real = [sizes_map[i] for i in size_dict]
        cost = cost1 + cost2 + opt_einsum.paths.compute_size_by_dict(i1_union_i2, size_dict_real)

        # Check the real cost
        if cost <= cost_cap:
            s = s1 | s2
            if s not in xn or cost < xn[s][1]:
                i_mem = opt_einsum.paths._dp_calc_legs(
                    g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2
                )

                # Get the memory scaling
                scaling = {}
                for i in i_mem:
                    c = dummy_map[size_dict[i]]
                    scaling[c] = scaling.get(c, 0) + 1

                # Check the memory scaling
                if max_ram_scaling is not None:
                    for cs, n in max_ram_scaling.items():
                        if sum(scaling.get(c, 0) for c in cs) > n:
                            return

                # Get the real memory
                mem = opt_einsum.paths.compute_size_by_dict(i_mem, size_dict_real)

                # Check the real memory
                if memory_limit is None or mem <= memory_limit:
                    # Accept this contraction
                    xn[s] = (i_mem, cost, (contract1, contract2))

    # Optimise the contraction path
    optimizer = opt_einsum.DynamicProgramming(minimize=cost)
    inputs, output, _sizes = _get_inputs_and_output(expr, sizes_dummy)
    _, _, index_sizes = _get_inputs_and_output(expr, sizes)
    path = optimizer(inputs, output, _sizes)
    tree = cotengra.ContractionTree.from_path(inputs, output, index_sizes, path=path)

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
    num_tensors = len(list(expr.search(Tensor)))
    if num_tensors < 2:
        return
    if num_tensors > 5:
        warnings.warn(
            "Generating all possible paths scales extremely poorly. Use with caution.",
            RuntimeWarning,
            stacklevel=2,
        )
    if sizes is None:
        sizes = _default_sizes

    def _recurse(path: list[T], n: int) -> Generator[list[T], None, None]:
        if n == 1:
            yield list(path)
            return
        for i, j in itertools.combinations(range(n), 2):
            path.append((i, j))
            yield from _recurse(path, n - 1)
            path.pop()

    # Get all paths
    inputs, output, index_sizes = _get_inputs_and_output(expr, sizes)
    for path in _recurse([], num_tensors):
        tree = cotengra.ContractionTree.from_path(inputs, output, index_sizes, path=path)
        yield tree


def generate_paths_approximate(
    expr: Mul,
    sizes: Optional[dict[str | None, int]] = None,
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
    num_tensors = len(list(expr.search(Tensor)))
    if num_tensors < 2:
        return
    if sizes is None:
        sizes = _default_sizes

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
    num_tensors = len(list(expr.search(Tensor)))
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
