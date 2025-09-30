"""Tools for optimisation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from albert import _default_sizes
from albert.canon import canonicalise_indices
from albert.expression import Expression
from albert.index import Index
from albert.scalar import Scalar
from albert.symmetry import Permutation, Symmetry
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Optional

    from albert.base import Base

    TensorInfo = tuple[str, tuple[str | None, ...], tuple[str | None, ...]]


def substitute_expressions(exprs: list[Expression]) -> list[Expression]:
    """Substitute expressions resulting from common subexpression elimination.

    Args:
        exprs: The tensor expressions.

    Returns:
        The total expression with the substituted tensors, for each distinct output tensor.
    """
    expr = [Expression(e.lhs, e.rhs.expand()) for e in exprs]

    # Find original set of indices for canonicalisation
    extra_indices: set[Index] = set()
    for e in expr:
        for tensor in e.rhs.search_leaves(Tensor):
            extra_indices.update(tensor.external_indices)
            extra_indices.update(tensor.internal_indices)
    extra_indices = sorted(extra_indices)

    memo = dict(found=False, counter=0)
    while True:
        # Find a valid substitution
        memo["found"] = False
        for j, e_src in enumerate(expr):
            for i, e_dst in enumerate(expr):
                if i == j:
                    continue

                def _substitute(tensor: Tensor) -> Base:
                    """Perform the substitution."""
                    if e_src.lhs.name == tensor.name:  # noqa: B023
                        memo["found"] = True

                        # Find the index substitution
                        index_map = dict(
                            zip(e_src.external_indices, tensor.external_indices)  # noqa: B023
                        )
                        for tensor in e_src.rhs.search_leaves(Tensor):  # noqa: B023
                            for index in tensor.external_indices:
                                if index not in index_map:
                                    index_map[index] = index.copy(name=f"z{memo['counter']}_")
                                    memo["counter"] += 1

                        # Substitute the expression
                        return e_src.rhs.map_indices(index_map)  # noqa: B023

                    return tensor

                expr[i] = Expression(e_dst.lhs, e_dst.rhs.apply(_substitute, Tensor))

            # Check if we found a substitution
            if memo["found"]:
                expr.pop(j)
                break

        # Check if we are done
        if not memo["found"]:
            break

    # Canonicalise the indices
    for i, e in enumerate(expr):
        perm = [e.rhs.external_indices.index(index) for index in e.external_indices]
        expr_i = canonicalise_indices(e.rhs, extra_indices=extra_indices)
        output_i = e.lhs.copy(*[expr_i.external_indices[p] for p in perm])
        expr[i] = Expression(output_i, expr_i)

    return expr


def combine_expressions(exprs: list[Expression]) -> list[Expression]:
    """Combine identical expressions.

    Args:
        exprs: The tensor expressions.

    Returns:
        The total expression with the combined tensors, for each distinct output tensor.
    """
    # Get the factors of the unique expressions
    expr_factors: dict[Expression, Scalar] = defaultdict(lambda: Scalar(0.0))
    for expr in exprs:
        for mul in expr.rhs.expand()._children:
            factor = Scalar(1.0)
            tensors = []
            for leaf in mul._children:
                if isinstance(leaf, Scalar):
                    factor *= leaf
                else:
                    tensors.append(leaf)
            mul_no_factor = mul.copy(*tensors)
            expr_factors[Expression(expr.lhs, mul_no_factor)] += factor

    # Find the unique expressions
    exprs: list[Expression] = []
    for expr, factor in expr_factors.items():
        exprs.append(Expression(expr.lhs, factor * expr.rhs))

    return exprs


def _tensor_info(tensor: Tensor) -> TensorInfo:
    """Get the information of a tensor (used as a node in a graph)."""
    return (
        tensor.name,
        tuple(i.spin for i in tensor.external_indices),
        tuple(i.space for i in tensor.external_indices),
    )


def expressions_to_graph(exprs: list[Expression]) -> dict[TensorInfo, set[TensorInfo]]:
    """Convert the expressions into a graph.

    Args:
        exprs: The tensor expressions.
    """
    graph: dict[TensorInfo, set[TensorInfo]] = defaultdict(set)
    for expr in exprs:
        info = _tensor_info(expr.lhs)
        for tensor in expr.rhs.search_leaves(Tensor):
            graph[info].add(_tensor_info(tensor))
    return graph


def split_expressions(exprs: list[Expression]) -> list[Expression]:
    """Split the expressions into single contractions.

    Args:
        exprs: The tensor expressions.

    Returns:
        The tensor expressions split up into single contractions.
    """
    new_exprs: list[Expression] = []
    for expr in exprs:
        for child in expr.rhs.expand()._children:
            new_exprs.append(Expression(expr.lhs, child))
    return new_exprs


# FIXME: Should be possible to improve this algorithm
def sort_expressions(exprs: list[Expression]) -> list[Expression]:
    """Sort expression to optimise intermediate tensor memory footprint.

    This is basically a subset of the topological sort algorithm, but with some additional
    ordering constraints.

    Args:
        exprs: The tensor expressions.

    Returns:
        The tensor expressions sorted to optimise intermediate tensor memory footprint.
    """
    import networkx

    # Get a dictionary of the expressions
    names: dict[TensorInfo, list[Expression]] = defaultdict(list)
    for expr in split_expressions(exprs):
        names[_tensor_info(expr.lhs)].append(expr)

    # Create a graph
    graph = expressions_to_graph(exprs)

    # Define the function to sort the names
    new_exprs: list[Expression] = []

    def _add(name: TensorInfo) -> None:
        """Add an expression to the list."""
        # Find the first time the tensor is used
        for i, expr in enumerate(new_exprs):
            if name in set(_tensor_info(t) for t in expr.rhs.search_leaves(Tensor)):
                break
        else:
            i = len(new_exprs)

        # Insert the expression
        if name in names:
            for expr in names[name]:
                new_exprs.insert(i, expr)
                i += 1

    # Define a function to recursively get the dependencies
    _cache: dict[TensorInfo, set[TensorInfo]] = {}

    def _get_deps(name: TensorInfo) -> set[TensorInfo]:
        """Get the dependencies of a tensor."""
        if name not in _cache:
            # Get the dependencies
            deps: set[TensorInfo] = set()
            for dep in graph[name]:
                if dep in graph:
                    deps.add(dep)
                deps.update(_get_deps(dep))

            # Cache the result
            _cache[name] = deps

        return _cache[name]

    # Sort the names
    for group in list(networkx.topological_generations(networkx.DiGraph(graph))):
        for name in sorted(group, key=lambda x: len(_get_deps(x))):
            _add(name)

    # Check the sanity of the result
    # FIXME: Doesn't really have to be true...
    # assert len(outputs) == len(exprs) == len(output_expr)

    return new_exprs


def count_flops(expr: Base, sizes: Optional[dict[str | None, int]] = None) -> int:
    """Count the number of FLOPs required to evaluate an expression.

    Args:
        expr: The expression to count the FLOPs of.
        sizes: The sizes of the spaces in the expression.

    Returns:
        The number of FLOPs required to evaluate the expression.
    """
    if sizes is None:
        sizes = _default_sizes

    # Find the FLOPs of the current expression
    flops = 1
    for index in expr.external_indices:
        flops *= sizes[index.space]
    for index in expr.internal_indices:
        flops *= sizes[index.space]

    # Add the FLOPs of the children recursively
    if expr._children:
        for child in expr._children:
            if isinstance(child, Tensor):
                flops += count_flops(child, sizes)

    return flops


def optimise_eom(
    returns: list[Tensor],
    exprs: list[Expression],
    method: str = "auto",
    **kwargs: Any,
) -> tuple[tuple[list[Tensor], list[Expression]], tuple[list[Tensor], list[Expression]]]:
    """Perform common subexpression elimination for EOM expressions.

    This function optimises out expressions that are independent of the EOM vectors.

    Args:
        returns: The return tensors.
        exprs: The tensor expressions to be optimised.
        method: The optimisation method to use. Options are `"auto"`, `"gristmill"`.
        **kwargs: Additional keyword arguments to pass to the optimiser.

    Returns:
        The returned tensors and optimised tensor expressions, split into those that depend on the
        EOM vectors and those that do not.
    """
    from albert.opt import optimise

    def _is_eom_vector(tensor: Tensor) -> bool:
        """Check if a tensor is an EOM vector."""
        return tensor.name.startswith("r") or tensor.name.startswith("l")

    # Track the classes
    classes: dict[str, type[Tensor]] = {}

    def _add_dummy_index(tensor: Tensor) -> Tensor:
        """Add a dummy index to EOM vector tensors."""
        if tensor.name not in classes:
            classes[tensor.name] = tensor.__class__
        indices = tensor.external_indices
        symmetry = tensor.symmetry
        if _is_eom_vector(tensor):
            indices = (Index("DUMMY", space="d"), *indices)
            symmetry = (
                Symmetry(*[Permutation((0,), 1) + perm for perm in symmetry.permutations])
                if symmetry
                else None
            )
        return Tensor(*indices, name=tensor.name, symmetry=symmetry)

    def _replace_types(tensor: Tensor) -> Tensor:
        """Replace the tensor types."""
        cls = classes.get(tensor.name, Tensor)
        return cls(*tensor.external_indices, symmetry=tensor.symmetry, name=tensor.name)

    def _remove_dummy_index(tensor: Tensor) -> Tensor:
        """Remove the dummy index from EOM vector tensors."""
        remove = [ind.space == "d" for ind in tensor.external_indices]
        indices = tuple(ind for ind, rem in zip(tensor.external_indices, remove) if not rem)
        symmetry = (
            Symmetry(
                *[
                    Permutation(
                        tuple(
                            i - sum(remove) for j, i in enumerate(perm.permutation) if not remove[j]
                        ),
                        perm.sign,
                    )
                    for perm in tensor.symmetry.permutations
                ]
            )
            if tensor.symmetry
            else None
        )
        return tensor.copy(*indices, symmetry=symmetry)

    # Make the optimiser more likely to optimise out constant intermediate tensors
    for i, expr in enumerate(exprs):
        exprs[i] = Expression(_add_dummy_index(expr.lhs), expr.rhs.apply(_add_dummy_index, Tensor))

    # Optimise with the dummy indices
    exprs = optimise(exprs, method, **kwargs)

    # Remove the dummy indices
    for i, expr in enumerate(exprs):
        exprs[i] = Expression(
            _remove_dummy_index(expr.lhs), expr.rhs.apply(_remove_dummy_index, Tensor)
        )

    # Extract the intermediates that don't depend on the EOM vectors
    exprs_dep: list[Expression] = []
    exprs_indep: list[Expression] = []
    cache: set[str] = set()
    for expr in exprs:
        depends = _is_eom_vector(expr.lhs)
        if not depends:
            for tensor in expr.rhs.search_leaves(Tensor):
                if _is_eom_vector(tensor) or tensor.name in cache:
                    depends = True
                    break
        if depends:
            exprs_dep.append(expr)
            cache.add(expr.lhs.name)
        else:
            exprs_indep.append(expr)

    # Get the intermediates needed to return
    returns_dep = returns
    returns_indep: list[Tensor] = []
    initialised: set[str] = set()
    for expr in exprs_dep:
        if expr.lhs.name.startswith("tmp"):
            initialised.add(expr.lhs.name)
        for tensor in expr.rhs.search_leaves(Tensor):
            if tensor.name.startswith("tmp") and tensor.name not in initialised:
                returns_indep.append(tensor)

    # Transform the names of the intermediates
    for i, expr in enumerate(exprs_dep):
        rhs = expr.rhs.apply(
            lambda t: (
                t.copy(name=f"ints.{t.name}")
                if t.name.startswith("tmp") and t.name not in initialised
                else t
            ),
            Tensor,
        )
        exprs_dep[i] = Expression(expr.lhs, rhs)

    # Re-optimise the output
    exprs_dep: list[Expression] = [Expression(expr.lhs, expr.rhs.expand()) for expr in exprs_dep]
    exprs_dep = substitute_expressions(exprs_dep)
    exprs_dep = optimise(exprs_dep, method, **kwargs)

    # Replace the tensor types
    for i, expr in enumerate(exprs_dep):
        exprs_dep[i] = Expression(_replace_types(expr.lhs), expr.rhs.apply(_replace_types, Tensor))
    for i, expr in enumerate(exprs_indep):
        exprs_indep[i] = Expression(
            _replace_types(expr.lhs), expr.rhs.apply(_replace_types, Tensor)
        )
    returns_dep = [_replace_types(tensor) for tensor in returns_dep]
    returns_indep = [_replace_types(tensor) for tensor in returns_indep]

    return (returns_indep, exprs_indep), (returns_dep, exprs_dep)
