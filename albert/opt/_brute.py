"""Brute-force common subexpression elimination.

This is deprecated and `albert.opt.cse` is recommended instead.
"""

from __future__ import annotations

import functools
import itertools
import warnings
from typing import TYPE_CHECKING, cast

from albert import _default_sizes
from albert.algebra import Add, Algebraic, Mul
from albert.canon import canonicalise_indices
from albert.expression import Expression
from albert.opt.tools import count_flops, sort_expressions
from albert.scalar import Scalar
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Optional

    from albert.base import Base
    from albert.index import Index


@functools.lru_cache(maxsize=512)
def _count_tensors(expr: Base) -> int:
    """Count the number of tensors in an expression."""
    if not expr._children:
        return 0
    count = 0
    for tensor in expr._children:
        if isinstance(tensor, Tensor):
            count += 1
        else:
            count += _count_tensors(tensor)
    return count


def _count_scalars(expr: Base) -> int:
    """Count the number of scalars in an expression."""
    if not expr._children:
        return 0
    count = 0
    for scalar in expr._children:
        if isinstance(scalar, Scalar):
            count += 1
        else:
            count += _count_scalars(scalar)
    return count


def _identify_subexpressions(
    exprs: list[Expression], indices: Optional[set[Index]] = None
) -> dict[tuple[Base, tuple[Index, ...]], int]:
    """Identify candidate common subexpressions and count their occurrences."""
    if indices is None:
        indices = set()
        for expr in exprs:
            for tensor in expr.rhs.search(Tensor):
                indices.update(set(tensor.indices))

    candidates: dict[tuple[Base, tuple[Index, ...]], int] = {}
    for expr in exprs:
        for mul in expr.rhs.search(Mul):
            # Loop over all combinations of >1 children to find subexpressions
            children = [child for child in mul._children if not isinstance(child, Scalar)]
            for r in range(2, len(children) + 1):
                for combo in itertools.combinations(children, r):
                    # Get the candidate subexpression
                    candidate: Base = Mul(*combo)

                    # Find the external indices of the candidate -- for Einstein summation
                    # compliant expressions, this is just candidate.external_indices, but we want
                    # to support more general expressions
                    other_indices = set()
                    for child in children:
                        if child not in combo:
                            other_indices.update(set(child.external_indices))
                            other_indices.update(set(child.internal_indices))
                    other_indices.update(set(expr.lhs.indices))
                    candidate_indices = set(candidate.external_indices + candidate.internal_indices)
                    candidate_indices = set.intersection(candidate_indices, other_indices)

                    # Canonicalise the candidate
                    index_map = _get_canonicalise_intermediate_map(candidate, indices)
                    _, candidate = _canonicalise_intermediate(None, candidate, indices)
                    canon_indices = tuple(index_map[i] for i in candidate_indices)

                    # Increment the count for this candidate
                    candidates[candidate, canon_indices] = (
                        candidates.get((candidate, canon_indices), 0) + 1
                    )

    return candidates


def parenthesise_mul(
    mul: Mul,
    sizes: Optional[dict[str | None, int]] = None,
    scaling_limit_cpu: dict[tuple[str, ...], int] | None = None,
    scaling_limit_ram: dict[tuple[str, ...], int] | None = None,
    intermediate_counter: int = 0,
) -> tuple[Mul, list[Expression]]:
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
        The parenthesised contraction represented by a non-nested product, and a list of tensor
        expressions defining the intermediates to resolve the nested product.
    """
    import opt_einsum

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
    tensors = list(mul.search(Tensor, depth=1))
    scalars = list(mul.search(Scalar, depth=1))
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
    intermediates: list[Expression] = []
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
            intermediates.append(Expression(interm, Mul(*tensors_i)))
            tensors.append(interm)

    return expr, intermediates


def factorise(exprs: list[Expression]) -> list[Expression]:
    """Factorise expressions that differ by at most one tensor and the scalar factor.

    Args:
        exprs: The tensor expressions to identify common subexpressions in.

    Returns:
        The factorised tensor expressions.
    """
    # Check that each expression is either:
    #  a) a Mul with at most two non-scalar children
    #  b) a non-scalar
    new_exprs: list[Expression] = []
    to_factorise: list[Expression] = []
    for expr in exprs:
        if isinstance(expr.rhs, Mul):
            children = [child for child in expr.rhs._children if not isinstance(child, Scalar)]
            if len(children) > 2:
                raise ValueError(
                    "Each expression must be a Mul with two non-scalar children. Try "
                    "parenthesising the expressions first.",
                )
            if len(children) == 2:
                to_factorise.append(expr)
            else:
                new_exprs.append(expr)
        else:
            new_exprs.append(expr)

    while to_factorise:
        # Get all the possible factors
        factors: dict[Base, int] = {}
        for expr in to_factorise:
            assert expr.rhs._children is not None
            children = [child for child in expr.rhs._children if not isinstance(child, Scalar)]
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
        for expr in to_factorise:
            if factor in expr.rhs.children:
                group.append((expr.lhs, Mul(*[child for child in expr.rhs.children if child != factor])))
            else:
                new_to_factorise.append((expr.lhs, expr.rhs))
        to_factorise = new_to_factorise

        # Combine the group into sums for each unique output
        for output in set(output for output, _ in group):
            group_out = [child for out, child in group if out == output]
            new_exprs.append(Expression(output, Mul(factor, Add(*group_out))))

    return new_exprs


def eliminate_common_subexpressions(
    exprs: list[Expression], sizes: Optional[dict[str | None, int]] = None
) -> list[Expression]:
    """Identify common subexpressions in a series of expressions.

    Expression should be parenthesised and split into individual contractions for this to work.

    Args:
        exprs: The tensor expressions to identify common subexpressions in.
        sizes: The sizes of the spaces in the expressions.

    Returns:
        Expressions with common subexpressions eliminated, and a list of intermediate definitions.
    """
    if sizes is None:
        sizes = _default_sizes

    # Get all indices in the expressions
    indices: set[Index] = set()
    for expr in exprs:
        for tensor in expr.rhs.search(Tensor):
            indices.update(set(tensor.indices))

    # Check if there are any existing intermediates we should avoid clashing with
    counter = 0
    for expr in exprs:
        for tensor in itertools.chain([expr.lhs], expr.rhs.search(Tensor)):
            if tensor.name.startswith("tmp") and tensor.name[3:].isdigit():
                counter = max(counter, int(tensor.name[3:]) + 1)

    while True:
        # Find candidate subexpressions and count their occurrences
        # TODO: write update function to avoid repeating work
        candidates = _identify_subexpressions(exprs, indices=indices)
        candidates = {k: v for k, v in candidates.items() if v > 1}

        # If no candidates, we're done
        if not candidates:
            break

        def _cost(c: tuple[Base, tuple[Index, ...]]) -> float:
            """Estimate the cost (benefit) of a candidate intermediate."""
            count = candidates[c]  # noqa: B023
            flops = count_flops(c[0], sizes=sizes)
            return count * flops

        # Favour the best candidate according to the cost function
        candidate, candidate_indices = max(candidates, key=_cost)

        # Initialise the intermediate
        interm = Tensor(
            *candidate_indices,
            name=f"tmp{counter}",
        )

        # Find all instances of the candidate
        # TODO: track addresses when searching for candidates to avoid repeating work
        new_exprs: list[Expression] = []
        touched = False
        for i, expr in enumerate(exprs):
            # Find the substitutions
            substs: dict[Base, Base] = {}
            for mul in expr.rhs.search(Mul):
                # Loop over combinations of children to find subexpressions
                children = [child for child in mul._children if not isinstance(child, Scalar)]
                assert candidate._children is not None
                for combo in itertools.combinations(children, len(candidate._children)):
                    mul_check = Mul(*combo)
                    _, mul_check_canon = _canonicalise_intermediate(None, mul_check, indices)
                    if mul_check_canon == candidate:
                        index_map = _get_canonicalise_intermediate_map(mul_check, indices)
                        index_map_rev = {v: k for k, v in index_map.items()}
                        scalars = [child for child in mul._children if child not in combo]
                        substs[mul] = Mul.factory(*scalars, interm.map_indices(index_map_rev))
                        touched = True

            if substs:
                # Apply the substitutions
                new_expr = expr.rhs.apply(lambda node: substs.get(node, node), Mul)  # noqa: B023
                new_exprs.append(Expression(expr.lhs, new_expr))
            else:
                new_exprs.append(expr)

        exprs = new_exprs

        if touched:
            # Add the definition of the intermediate and increment the counter
            exprs.append(Expression(interm, candidate))
            counter += 1

    # For any remaining nested multiplications, assign intermediates instead of the nesting
    new_exprs = []

    def _separate(mul: Mul) -> Mul:
        """Separate a nested multiplication."""
        nonlocal counter

        children: list[Base] = []
        for child in mul._children:
            if isinstance(child, Algebraic):
                # Create an intermediate for this nested multiplication
                intermediate = Tensor(
                    *child.external_indices,
                    name=f"tmp{counter}",
                )
                counter += 1
                exprs.append(Expression(intermediate, child))
                children.append(intermediate)
            else:
                children.append(child)

        return Mul(*children)

    for expr in exprs:
        new_exprs.append(Expression(expr.lhs, expr.rhs.apply(_separate, Mul)))
    exprs = new_exprs

    return exprs


def absorb_intermediate_factors(exprs: list[Expression]) -> list[Expression]:
    """Absorb factors from intermediates back into the expressions where possible.

    Args:
        exprs: The tensor expressions to update.

    Returns:
        The updated tensor expressions.
    """
    new_exprs: list[Expression] = []
    for i, expr in enumerate(exprs):
        if not expr.lhs.name.startswith("tmp"):
            new_exprs.append(expr)
            continue
        scalars = list(filter(lambda child: isinstance(child, Scalar), expr.rhs._children or []))
        others = list(filter(lambda child: not isinstance(child, Scalar), expr.rhs._children or []))
        if len(others) == len(scalars) == 1:
            for j, ex in enumerate(new_exprs):
                new_exprs[j] = Expression(
                    ex.lhs,
                    ex.rhs.apply(
                        lambda node: (
                            Mul(*scalars, node) if node.name == ex.lhs.name else node  # noqa: B023
                        ),
                        Tensor,
                    ),
                )
            new_exprs.append(Expression(expr.lhs, others[0]))
        else:
            new_exprs.append(expr)
    return new_exprs


def merge_identical_intermediates(exprs: list[Expression]) -> list[Expression]:
    """Merge identical intermediates to avoid duplication.

    Args:
        exprs: The tensor expressions to update.

    Returns:
        The updated tensor expressions.
    """
    # TODO: relax the identical indices requirement to allow for transposes
    groups: dict[tuple[Base, tuple[Index, ...]], list[Tensor]] = {}
    for expr in exprs:
        if (expr.rhs, expr.lhs.indices) not in groups:
            groups[expr.rhs, expr.lhs.indices] = []
        groups[expr.rhs, expr.lhs.indices].append(expr.lhs)
    unique_intermediates: dict[str, Tensor] = {}
    for _, outputs in groups.items():
        for output in outputs:
            unique_intermediates[output.name] = outputs[0]

    def _apply(node: Tensor) -> Tensor:
        if node.name.startswith("tmp"):
            return node.__class__(*node.indices, name=unique_intermediates[node.name].name)
        return node

    return [
        Expression(expr.lhs, expr.rhs.apply(_apply, Tensor))
        for expr in exprs
        if expr.lhs.name == unique_intermediates[expr.lhs.name].name
    ]


def absorb_trivial_intermediates(exprs: list[Expression]) -> list[Expression]:
    """Absorb intermediates that are just a single tensor back into the expressions.

    Args:
        exprs: The tensor expressions to update.

    Returns:
        The updated expression.
    """
    trivial: dict[str, bool] = {}
    definitions: dict[str, Expression] = {}
    for i, expr in enumerate(exprs):
        if expr.lhs.name.startswith("tmp") and isinstance(expr.rhs, Tensor):
            # If the output has multiple single tensor expressions, it's not trivial
            trivial[expr.lhs.name] = expr.lhs.name not in trivial and True
            definitions[expr.lhs.name] = expr

    def _apply(node: Tensor) -> Tensor:
        while trivial.get(node.name, False):
            expr = definitions[node.name]
            index_map = dict(zip(expr.lhs.indices, node.indices))
            node = expr.rhs.map_indices(index_map)  # type: ignore[assignment]
        return node

    return [
        Expression(expr.lhs, expr.rhs.apply(_apply, Tensor))
        for expr in exprs
        if not trivial.get(expr.lhs.name, False)
    ]


def unused_intermediates(exprs: list[Expression]) -> list[Tensor]:
    """Identify intermediates that are defined but not used.

    Args:
        exprs: The tensor expressions to check.

    Returns:
        The list of unused intermediate tensors.
    """
    defined: set[Tensor] = set()
    used: set[str] = set()
    for expr in exprs:
        if expr.lhs.name.startswith("tmp"):
            defined.add(expr.lhs)
        for tensor in expr.rhs.search(Tensor):
            if tensor.name.startswith("tmp"):
                used.add(tensor.name)
    return [tensor for tensor in defined if tensor.name not in used]


def undefined_intermediates(exprs: list[Expression]) -> list[Tensor]:
    """Identify intermediates that are used but not defined.

    Args:
        exprs: The tensor expressions to check.

    Returns:
        The list of undefined intermediate tensors.
    """
    defined: set[str] = set()
    used: set[Tensor] = set()
    for expr in exprs:
        if expr.lhs.name.startswith("tmp"):
            defined.add(expr.lhs.name)
        for tensor in expr.rhs.search(Tensor):
            if tensor.name.startswith("tmp"):
                used.add(tensor)
    return [tensor for tensor in used if tensor.name not in defined]


def renumber_intermediates(exprs: list[Expression]) -> list[Expression]:
    """Renumber intermediates to ensure a contiguous sequence.

    Args:
        exprs: The tensor expressions to renumber.

    Returns:
        The renumbered tensor expressions.
    """
    # Sort the expressions so the renumbering looks sensible after code generation
    exprs = sort_expressions(exprs)

    # Map old intermediate names to new ones
    counter = 0
    name_map: dict[str, str] = {}
    for expr in exprs:
        for tensor in itertools.chain([expr.lhs], expr.rhs.search(Tensor)):
            if tensor.name.startswith("tmp"):
                if tensor.name not in name_map:
                    name_map[tensor.name] = f"tmp{counter}"
                    counter += 1

    def _apply(node: Tensor) -> Tensor:
        if node.name.startswith("tmp"):
            return node.__class__(*node.indices, name=name_map[node.name])
        return node

    exprs = [
        Expression(
            expr.lhs.__class__(*expr.lhs.indices, name=name_map.get(expr.lhs.name, expr.lhs.name)),
            expr.rhs.apply(_apply, Tensor),
        )
        for expr in exprs
    ]

    return exprs


@functools.lru_cache(maxsize=32)
def _get_index_groups(
    indices: frozenset[Index],
) -> dict[tuple[str | None, str | None], list[Index]]:
    """Group indices by their (space, spin) pairs."""
    index_groups: dict[tuple[str | None, str | None], list[Index]] = {}
    for index in indices:
        key = (index.space, index.spin)
        if key not in index_groups:
            index_groups[key] = []
        index_groups[key].append(index)
    return {key: sorted(value) for key, value in index_groups.items()}


def _get_canonicalise_intermediate_map(expr: Base, indices: set[Index]) -> dict[Index, Index]:
    """Get the index mapping to canonicalise the indices of an intermediate."""
    index_groups = _get_index_groups(frozenset(indices))
    indices_i = {
        key: [
            index
            for index in (expr.external_indices + expr.internal_indices)
            if (index.space, index.spin) == key
        ]
        for key in index_groups
    }
    index_map = {}
    for key in index_groups:
        for old, new in zip(index_groups[key], indices_i[key]):
            index_map[new] = old
    return index_map


def _canonicalise_intermediate(
    output: Tensor | None, expr: Base, indices: set[Index]
) -> tuple[Tensor, Base]:
    """Canonicalise the indices of an intermediate."""
    index_map = _get_canonicalise_intermediate_map(expr, indices)
    expr = expr.map_indices(index_map)
    output = output.map_indices(index_map) if output is not None else None
    return output, expr  # type: ignore[return-value]


def eliminate_and_factorise_common_subexpressions(
    expr: Expression,
    sizes: Optional[dict[str | None, int]] = None,
    scaling_limit_cpu: dict[tuple[str, ...], int] | None = None,
    scaling_limit_ram: dict[tuple[str, ...], int] | None = None,
    max_passes: int = 3,
) -> list[Expression]:
    """Identify common subexpressions in an expression, with parenthesisation and factorisation.

    Expression should be canonicalised for this to work well.

    Args:
        expr: The tensor expression to identify common subexpressions in.
        sizes: The sizes of the spaces in the expression.
        scaling_limit_cpu: The scaling limits for CPU. Keys should be tuples of index space names,
            and values are the maximum allowed scaling for that combination of spaces.
        scaling_limit_ram: The scaling limits for RAM. Keys should be tuples of index space names,
            and values are the maximum allowed scaling for that combination of spaces.
        max_passes: The maximum number of passes to perform. More passes may find more common
            subexpressions, but will take longer.

    Returns:
        List of tensor expressions, which may correspond to the original output or intermediates.
    """
    # Collect all indices in the expression
    indices: set[Index] = set()
    for tensor in expr.rhs.search(Tensor):
        indices.update(set(tensor.indices))

    def _canonicalise(exprs: list[Expression]) -> list[Expression]:
        """Canonicalise the indices."""
        for i, expr in enumerate(exprs):
            if expr.lhs.name.startswith("tmp"):
                lhs, rhs = _canonicalise_intermediate(expr.lhs, expr.rhs, indices)
                expr = Expression(lhs, rhs)
            else:
                rhs = canonicalise_indices(expr.rhs, extra_indices=list(indices), which="internal")
                lhs = expr.lhs.map_indices(
                    dict(zip(expr.rhs.external_indices, rhs.external_indices))
                )
            exprs[i] = Expression(expr.lhs, expr.rhs.squeeze().canonicalise())
        return exprs

    # Parenthesise each multiplication
    exprs: list[Expression] = []
    counter = 0
    for mul in expr.rhs.expand().children:
        rhs, ints = parenthesise_mul(
            cast(Mul, mul),
            sizes=sizes,
            scaling_limit_cpu=scaling_limit_cpu,
            scaling_limit_ram=scaling_limit_ram,
            intermediate_counter=counter,
        )
        exprs.extend(ints)
        exprs.append(Expression(expr.lhs, rhs))
        counter += len(ints)

    # Eliminate common subexpressions
    for i in range(max_passes):
        exprs_prev = exprs.copy()
        if i != 0:
            exprs = factorise(exprs)
        exprs = eliminate_common_subexpressions(exprs, sizes=sizes)
        exprs = _canonicalise(exprs)
        exprs = absorb_trivial_intermediates(exprs)
        exprs = merge_identical_intermediates(exprs)
        if exprs == exprs_prev:
            break

    # Renumber intermediates, also sorts the expressions
    exprs = renumber_intermediates(exprs)

    unused = set(interm.name for interm in unused_intermediates(exprs))
    undefined = set(interm.name for interm in undefined_intermediates(exprs))
    if unused:
        warnings.warn(f"Intermediates defined but not used: {unused}.", stacklevel=2)
    if undefined:
        warnings.warn(f"Intermediates used but not defined: {undefined}.", stacklevel=2)

    return exprs
