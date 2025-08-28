"""Parenthesisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import opt_einsum

from albert import _default_sizes
from albert.algebra import Add, Mul
from albert.base import Base
from albert.index import Index
from albert.scalar import Scalar
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Optional


def parenthesise_mul(
    mul: Mul,
    sizes: Optional[dict[str | None, float]] = None,
    scaling_limit_cpu: dict[tuple[str, ...], int] | None = None,
    scaling_limit_ram: dict[tuple[str, ...], int] | None = None,
    intermediate_counter: int = 0,
) -> tuple[Mul, list[tuple[Tensor, Base]]]:
    """Parenthesise a product.

    Converts the `Mul` of given children into a nested `Mul` of groups of said children.

    Args:
        mul: The contraction to parenthesise.
        sizes: The sizes of the spaces in the expression.
        scaling_limit_cpu: The scaling limits for CPU. Keys should be tuples of index space names,
            and values are the maximum allowed scaling for that combination of spaces.
        scaling_limit_ram: The scaling limits for RAM. Keys should be tuples of index space names,
            and values are the maximum allowed scaling for that combination of spaces.
        intermediate_counter: The starting counter for naming intermediate tensors.

    Returns:
        The parenthesised contraction represented by a non-nested product, and a list of
        `(Tensor, Base)` pairs defining the intermediates to resolve the nested product.
    """
    if sizes is None:
        sizes = _default_sizes
    if scaling_limit_cpu is None:
        scaling_limit_cpu = {}
    if scaling_limit_ram is None:
        scaling_limit_ram = {}

    # Get dummy sizes for the cost function
    sizes_dummy = {space: ord(space) for space in sizes if isinstance(space, str)}
    dummy_map = {value: key for key, value in sizes_dummy.items()}
    sizes_map = {sizes_dummy[space]: sizes[space] for space in sizes_dummy}

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
        """Cost function for `opt_einsum`."""
        # Get the cost scaling
        scaling: dict[str, int] = {}
        for i in i1_union_i2:
            c = dummy_map[size_dict[i]]
            scaling[c] = scaling.get(c, 0) + 1

        # Check the cost scaling
        if scaling_limit_cpu is not None:
            for cs, n in scaling_limit_cpu.items():
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
                if scaling_limit_ram is not None:
                    for cs, n in scaling_limit_ram.items():
                        if sum(scaling.get(c, 0) for c in cs) > n:
                            return

                # Get the real memory
                mem = opt_einsum.paths.compute_size_by_dict(i_mem, size_dict_real)

                # Check the real memory
                if memory_limit is None or mem <= memory_limit:
                    # Accept this contraction
                    xn[s] = (i_mem, cost, (contract1, contract2))

    # Separate the children into tensors and scalars
    tensors = list(mul.search_children(Tensor))
    scalars = list(mul.search_children(Scalar))
    assert len(mul._children) == len(list(tensors)) + len(list(scalars))

    # Get the optimal contraction path
    optimiser = opt_einsum.DynamicProgramming(
        minimize=cost,
        cost_cap=True,
    )

    # Map index names to unique characters for opt_einsum
    _index_map: dict[Index, str] = {}

    def _assign_index(index: Index) -> str:
        if index not in _index_map:
            if len(_index_map) >= 26:
                raise ValueError("Too many unique indices.")
            _index_map[index] = chr(97 + len(_index_map))
        return _index_map[index]

    # Make fake arrays to get the contraction path
    arrays = [lambda: None for _ in tensors]
    for i, t in enumerate(tensors):
        arrays[i].shape = tuple(sizes_dummy[i.space] for i in t.indices)  # type: ignore
    inputs = ["".join(_assign_index(i) for i in t.indices) for t in tensors]
    output = "".join(_assign_index(i) for i in mul.external_indices)
    subscript = ",".join(inputs) + "->" + output
    path, info = opt_einsum.contract_path(subscript, *arrays, optimize=optimiser)
    lines = str(info).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("-----")) + 3
    subscripts = [line.split()[2] for line in lines[start:] if line.strip()]

    # Build the contractions
    intermediates: list[tuple[Tensor, Base]] = []
    counter = intermediate_counter
    _index_map_rev = {v: k for k, v in _index_map.items()}
    while subscripts:
        inputs_i, output_i = subscripts.pop(0).split("->")
        tensors_i = [tensors.pop(i) for i in sorted(path.pop(0), reverse=True)]
        assert all(
            tuple(_index_map_rev[c] for c in inp) == tuple(t.indices)
            for inp, t in zip(inputs_i.split(","), tensors_i)
        )
        if len(subscripts) == 0:
            expr = Mul(*scalars, *tensors_i)
        else:
            output_indices = [_index_map_rev[c] for c in output_i]
            interm = Tensor(*output_indices, name=f"tmp{counter}")
            counter += 1
            intermediates.append((interm, Mul(*tensors_i)))
            tensors.append(interm)

    return expr, intermediates


def factorise(output_exprs: list[tuple[Tensor, Base]]) -> list[tuple[Tensor, Base]]:
    """Factorise expressions that differ by at most one tensor and the scalar factor.

    Args:
        output_exprs: The output and expression pairs to identify common subexpressions in, as
            `(Tensor, Base)` pairs.

    Returns:
        The factorised expressions as `(Tensor, Base)` pairs.
    """
    # Check that each expression is either:
    #  a) a Mul with at most two non-scalar children
    #  b) a non-scalar
    new_output_exprs: list[tuple[Tensor, Base]] = []
    to_factorise: list[tuple[Tensor, Base]] = []
    for output, expr in output_exprs:
        if isinstance(expr, Mul):
            children = [child for child in expr._children if not isinstance(child, Scalar)]
            if len(children) > 2:
                raise ValueError(
                    "Each expression must be a Mul with two non-scalar children. Try "
                    "parenthesising the expressions first.",
                )
            if len(children) == 2:
                to_factorise.append((output, expr))
            else:
                new_output_exprs.append((output, expr))
        else:
            new_output_exprs.append((output, expr))

    while to_factorise:
        # Get all the possible factors
        factors: dict[Base, int] = {}
        for output, expr in to_factorise:
            assert expr._children is not None
            children = [child for child in expr._children if not isinstance(child, Scalar)]
            assert len(children) == 2
            for child in children:
                if child not in factors:
                    factors[child] = 0
                factors[child] += 1

        # Find the factor that appears the most
        factor = max(factors, key=lambda k: factors[k])

        # For each expression that contains this factor, remove it and group them
        group: list[tuple[Tensor, Base]] = []
        new_to_factorise: list[tuple[Tensor, Base]] = []
        for output, expr in to_factorise:
            assert expr._children is not None
            if factor in expr._children:
                group.append((output, Mul(*[child for child in expr._children if child != factor])))
            else:
                new_to_factorise.append((output, expr))
        to_factorise = new_to_factorise

        # Combine the group into sums for each unique output
        for output in set(output for output, _ in group):
            group_out = [child for out, child in group if out == output]
            new_output_exprs.append((output, Mul(factor, Add(*group_out))))

    return new_output_exprs
