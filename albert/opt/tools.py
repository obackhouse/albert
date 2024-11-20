"""Tools for optimisation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from albert import _default_sizes
from albert.canon import canonicalise_indices
from albert.index import Index
from albert.opt import optimise
from albert.symmetry import Permutation, Symmetry
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Optional

    from albert.base import Base

    TensorInfo = tuple[str, tuple[str | None, ...], tuple[str | None, ...]]


def substitute_expressions(output_expr: list[tuple[Tensor, Base]]) -> list[tuple[Tensor, Base]]:
    """Substitute expressions resulting from common subexpression elimination.

    The return value is a list of tuples, where the first element is the output tensor and the
    second element is the expression, for each distinct output tensor.

    Args:
        output_expr: The output tensors and their expressions.

    Returns:
        The total expression with the substituted tensors, for each distinct output tensor.
    """
    output = [out for out, _ in output_expr]
    expr: list[Base] = [exp.expand() for _, exp in output_expr]

    # Find original set of indices for canonicalisation
    extra_indices: set[Index] = set()
    for e in expr:
        for tensor in e.search_leaves(Tensor):
            extra_indices.update(tensor.external_indices)
            extra_indices.update(tensor.internal_indices)
    extra_indices = sorted(extra_indices)

    memo = dict(found=False, counter=0)
    while True:
        # Find a valid substitution
        memo["found"] = False
        for j, (o_src, e_src) in enumerate(zip(output, expr)):
            for i, e_dst in enumerate(expr):
                if i == j:
                    continue

                def _substitute(tensor: Tensor) -> Base:  # noqa: B023
                    """Perform the substitution."""
                    if o_src.name == tensor.name:
                        memo["found"] = True

                        # Find the index substitution
                        index_map = dict(zip(o_src.external_indices, tensor.external_indices))
                        for tensor in e_src.search_leaves(Tensor):
                            for index in tensor.external_indices:
                                if index not in index_map:
                                    index_map[index] = index.copy(name=f"z{memo['counter']}_")
                                    memo["counter"] += 1

                        # Substitute the expression
                        return e_src.map_indices(index_map)

                    return tensor

                expr[i] = e_dst.apply(_substitute, Tensor)

            # Check if we found a substitution
            if memo["found"]:
                output.pop(j)
                expr.pop(j)
                break

        # Check if we are done
        if not memo["found"]:
            break

    # Canonicalise the indices
    for i, (o, e) in enumerate(zip(output, expr)):
        perm = [e.external_indices.index(index) for index in o.external_indices]
        expr[i] = canonicalise_indices(e, extra_indices=extra_indices)
        output[i] = o.copy(*[expr[i].external_indices[p] for p in perm])

    return list(zip(output, expr))


def _tensor_info(tensor: Tensor) -> TensorInfo:
    """Get the information of a tensor (used as a node in a graph)."""
    return (
        tensor.name,
        tuple(i.spin for i in tensor.external_indices),
        tuple(i.space for i in tensor.external_indices),
    )


def expressions_to_graph(
    output_expr: list[tuple[Tensor, Base]]
) -> dict[TensorInfo, set[TensorInfo]]:
    """Convert the expressions into a graph.

    Args:
        output_expr: The output tensors and their expressions.
    """
    graph: dict[TensorInfo, set[TensorInfo]] = defaultdict(set)
    for output, expr in output_expr:
        info = _tensor_info(output)
        for tensor in expr.search_leaves(Tensor):
            graph[info].add(_tensor_info(tensor))
    return graph


def split_expressions(output_expr: list[tuple[Tensor, Base]]) -> list[tuple[Tensor, Base]]:
    """Split the expressions into single contractions.

    Args:
        output_expr: The output tensors and their expressions.

    Returns:
        The output tensors and their expressions split up into single contractions.
    """
    outputs: list[Tensor] = []
    exprs: list[Base] = []
    for output, expr in output_expr:
        for child in expr.expand()._children:
            outputs.append(output)
            exprs.append(child)
    return list(zip(outputs, exprs))


# FIXME: Should be possible to improve this algorithm
def sort_expressions(output_expr: list[tuple[Tensor, Base]]) -> list[tuple[Tensor, Base]]:
    """Sort expression to optimise intermediate tensor memory footprint.

    This is basically a subset of the topological sort algorithm, but with some additional
    ordering constraints.

    Args:
        output_expr: The output tensors and their expressions.

    Returns:
        The output tensors and their expressions sorted to optimise intermediate tensor memory
        footprint.
    """
    import networkx

    # Get a dictionary of the expressions
    names: dict[TensorInfo, list[tuple[Tensor, Base]]] = defaultdict(list)
    for output, expr in split_expressions(output_expr):
        names[_tensor_info(output)].append((output, expr))

    # Create a graph
    graph = expressions_to_graph(output_expr)

    # Define the function to sort the names
    outputs: list[Tensor] = []
    exprs: list[Base] = []

    def _add(name: TensorInfo) -> None:
        """Add an expression to the list."""
        # Find the first time the tensor is used
        for i, (output, expr) in enumerate(zip(outputs, exprs)):
            if name in set(_tensor_info(t) for t in expr.search_leaves(Tensor)):
                break
        else:
            i = len(outputs)

        # Insert the expression
        if name in names:
            for output, expr in names[name]:
                outputs.insert(i, output)
                exprs.insert(i, expr)
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
    assert len(outputs) == len(exprs) == len(output_expr)

    return list(zip(outputs, exprs))


def count_flops(expr: Base, sizes: Optional[dict[str | None, float]] = None) -> float:
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
    flops = 1.0
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
    outputs: list[Tensor],
    exprs: list[Base],
    method: str = "auto",
    **kwargs: Any,
) -> tuple[
    tuple[list[Tensor], list[tuple[Tensor, Base]]], tuple[list[Tensor], list[tuple[Tensor, Base]]]
]:
    """Perform common subexpression elimination for EOM expressions.

    This function optimises out expressions that are independent of the EOM vectors.

    Args:
        outputs: The output tensors for each expression.
        exprs: The expressions to be optimised.
        method: The optimisation method to use. Options are `"auto"`, `"gristmill"`.

    Returns:
        The optimised expressions, as tuples of the output tensor and the expression.
    """

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
    for i, (output, expr) in enumerate(zip(outputs, exprs)):
        outputs[i] = _add_dummy_index(output)
        exprs[i] = expr.apply(_add_dummy_index, Tensor)

    # Optimise with the dummy indices
    output_expr = optimise(outputs, exprs, method, **kwargs)

    # Remove the dummy indices
    for i, (output, expr) in enumerate(output_expr):
        output_expr[i] = (_remove_dummy_index(output), expr.apply(_remove_dummy_index, Tensor))

    # Extract the intermediates that don't depend on the EOM vectors
    output_expr_dep: list[tuple[Tensor, Base]] = []
    output_expr_indep: list[tuple[Tensor, Base]] = []
    cache: set[str] = set()
    for output, expr in output_expr:
        depends = _is_eom_vector(output)
        if not depends:
            for tensor in expr.search_leaves(Tensor):
                if _is_eom_vector(tensor) or tensor.name in cache:
                    depends = True
                    break
        if depends:
            output_expr_dep.append((output, expr))
            cache.add(output.name)
        else:
            output_expr_indep.append((output, expr))

    # Get the intermediates needed to return
    returns_dep = returns
    returns_indep: list[Tensor] = []
    initialised: set[str] = set()
    for output, expr in output_expr_dep:
        if output.name.startswith("tmp"):
            initialised.add(output.name)
        for tensor in expr.search_leaves(Tensor):
            if tensor.name.startswith("tmp") and tensor.name not in initialised:
                returns_indep.append(tensor)

    # Transform the names of the intermediates
    for i, (output, expr) in enumerate(output_expr_dep):
        expr = expr.apply(
            lambda t: (
                t.copy(name=f"ints.{t.name}")
                if t.name.startswith("tmp") and t.name not in initialised
                else t
            ),
            Tensor,
        )
        output_expr_dep[i] = (output, expr)

    # Re-optimise the output
    output_expr_dep: list[tuple[Tensor, Base]] = [
        (output, expr.expand()) for output, expr in output_expr_dep
    ]
    output_expr_dep = substitute_expressions(output_expr_dep)
    output_expr_dep = optimise(
        [output for output, _ in output_expr_dep],
        [expr for _, expr in output_expr_dep],
        method,
        **kwargs,
    )

    # Replace the tensor types
    for i, (output, expr) in enumerate(output_expr_dep):
        output_expr_dep[i] = (_replace_types(output), expr.apply(_replace_types, Tensor))
    for i, (output, expr) in enumerate(output_expr_indep):
        output_expr_indep[i] = (_replace_types(output), expr.apply(_replace_types, Tensor))
    returns_dep = [_replace_types(tensor) for tensor in returns_dep]
    returns_indep = [_replace_types(tensor) for tensor in returns_indep]

    return (returns_indep, output_expr_indep), (returns_dep, output_expr_dep)
