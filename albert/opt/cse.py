"""Common subexpression elimination and identification."""

from __future__ import annotations

import functools
import itertools
import warnings
from typing import TYPE_CHECKING

from albert import _default_sizes
from albert.algebra import Algebraic, Mul, _compose_mul
from albert.canon import canonicalise_indices
from albert.opt.parenth import factorise, parenthesise_mul
from albert.opt.tools import count_flops, sort_expressions
from albert.scalar import Scalar
from albert.tensor import Tensor
from albert.expression import Expression

if TYPE_CHECKING:
    from typing import Optional

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
        for _, expr in exprs:
            for tensor in expr.search_nodes(Tensor):
                indices.update(set(tensor.indices))

    candidates: dict[tuple[Base, tuple[Index, ...]], int] = {}
    for output, expr in exprs:
        for mul in expr.search_nodes(Mul):
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
                    other_indices.update(set(output.indices))
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


def eliminate_common_subexpressions(
    exprs: list[Expression], sizes: Optional[dict[str | None, float]] = None
) -> list[tuple[Tensor, Base]]:
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
    for _, expr in exprs:
        for tensor in expr.search_nodes(Tensor):
            indices.update(set(tensor.indices))

    # Check if there are any existing intermediates we should avoid clashing with
    counter = 0
    for output, expr in exprs:
        for tensor in itertools.chain([output], expr.search_nodes(Tensor)):
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
        new_exprs: list[tuple[Tensor, Base]] = []
        touched = False
        for i, (output, expr) in enumerate(exprs):
            # Find the substitutions
            substs: dict[Base, Base] = {}
            for mul in expr.search_nodes(Mul):
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
                        substs[mul] = _compose_mul(*scalars, interm.map_indices(index_map_rev))
                        touched = True

            if substs:
                # Apply the substitutions
                new_expr = expr.apply(lambda node: substs.get(node, node), Mul)  # noqa: B023
                new_exprs.append(Expression(output, new_expr))
            else:
                new_exprs.append(Expression(output, expr))

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

    for output, expr in exprs:
        new_exprs.append(Expression(output, expr.apply(_separate, Mul)))
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
    for i, (output, expr) in enumerate(exprs):
        if not output.name.startswith("tmp"):
            new_exprs.append(Expression(output, expr))
            continue
        scalars = list(filter(lambda child: isinstance(child, Scalar), expr._children or []))
        others = list(filter(lambda child: not isinstance(child, Scalar), expr._children or []))
        if len(others) == len(scalars) == 1:
            for j, (out, ex) in enumerate(new_exprs):
                new_exprs[j] = Expression(
                    out,
                    ex.apply(
                        lambda node: (
                            Mul(*scalars, node) if node.name == output.name else node  # noqa: B023
                        ),
                        Tensor,
                    ),
                )
            new_exprs.append(Expression(output, others[0]))
        else:
            new_exprs.append(Expression(output, expr))
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
    for output, expr in exprs:
        if (expr, output.indices) not in groups:
            groups[expr, output.indices] = []
        groups[expr, output.indices].append(output)
    unique_intermediates: dict[str, Tensor] = {}
    for _, outputs in groups.items():
        for output in outputs:
            unique_intermediates[output.name] = outputs[0]

    def _apply(node: Tensor) -> Tensor:
        if node.name.startswith("tmp"):
            return node.__class__(*node.indices, name=unique_intermediates[node.name].name)
        return node

    return [
        Expression(output, expr.apply(_apply, Tensor))
        for output, expr in exprs
        if output.name == unique_intermediates[output.name].name
    ]


def absorb_trivial_intermediates(exprs: list[Expression]) -> list[Expression]:
    """Absorb intermediates that are just a single tensor back into the expressions.

    Args:
        exprs: The tensor expressions to update.

    Returns:
        The updated expression.
    """
    trivial: dict[str, bool] = {}
    definitions: dict[str, tuple[Tensor, Base]] = {}
    for i, (output, expr) in enumerate(exprs):
        if output.name.startswith("tmp") and isinstance(expr, Tensor):
            # If the output has multiple single tensor expressions, it's not trivial
            trivial[output.name] = output.name not in trivial and True
            definitions[output.name] = (output, expr)

    def _apply(node: Tensor) -> Tensor:
        while trivial.get(node.name, False):
            output, expr = definitions[node.name]
            index_map = dict(zip(output.indices, node.indices))
            node = expr.map_indices(index_map)
        return node

    return [
        Expression(output, expr.apply(_apply, Tensor))
        for output, expr in exprs
        if not trivial.get(output.name, False)
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
    for output, expr in exprs:
        if output.name.startswith("tmp"):
            defined.add(output)
        for tensor in expr.search_nodes(Tensor):
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
    for output, expr in exprs:
        if output.name.startswith("tmp"):
            defined.add(output.name)
        for tensor in expr.search_nodes(Tensor):
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
    for output, expr in exprs:
        for tensor in itertools.chain([output], expr.search_nodes(Tensor)):
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
            output.__class__(*output.indices, name=name_map.get(output.name, output.name)),
            expr.apply(_apply, Tensor),
        )
        for output, expr in exprs
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
    sizes: Optional[dict[str | None, float]] = None,
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
    expression, output = expr

    # Collect all indices in the expression
    indices: set[Index] = set()
    for tensor in expression.search_nodes(Tensor):
        indices.update(set(tensor.indices))

    def _canonicalise(exprs: list[Expression]) -> list[Expression]:
        """Canonicalise the indices."""
        for i, (output, expr) in enumerate(exprs):
            if output.name.startswith("tmp"):
                output, expr = _canonicalise_intermediate(output, expr, indices)
            else:
                expr = canonicalise_indices(expr, extra_indices=list(indices), which="internal")
            expr = expr.squeeze().canonicalise()
            exprs[i] = Expression(output, expr)
        return exprs

    # Parenthesise each multiplication
    exprs: list[Expression] = []
    counter = 0
    for mul in expression.expand()._children:
        expression, ints = parenthesise_mul(
            mul,
            sizes=sizes,
            scaling_limit_cpu=scaling_limit_cpu,
            scaling_limit_ram=scaling_limit_ram,
            intermediate_counter=counter,
        )
        exprs.extend(ints)
        exprs.append(Expression(output, expression))
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
