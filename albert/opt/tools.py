"""Tools for optimisation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from albert.tensor import Tensor

if TYPE_CHECKING:
    from albert.base import Base

    TensorInfo = tuple[str, tuple[str | None, ...], tuple[str | None, ...]]


def substitute_expressions(output_expr: list[tuple[Tensor, Base]]) -> Base:
    """Substitute expressions resulting from common subexpression elimination.

    The outputs and expressions given should represent the result of the CSE on a single
    original expression, and therefore allow a complete substitution returning a single
    expression.

    Args:
        output_expr: The output tensors and their expressions.

    Returns:
        The total expression with the substituted tensors.

    Raises:
        ValueError: If a complete substitution is not possible.
    """
    output = [out for out, _ in output_expr]
    expr = [exp for _, exp in output_expr]

    idx_counter = 0
    while True:
        # Find a valid substitution
        found = False
        for i, e_dst in enumerate(expr):
            for j, (o_src, e_src) in enumerate(zip(output, expr)):
                if i == j:
                    continue

                for t_dst in e_dst.search_leaves(Tensor):
                    if o_src.name == t_dst.name:
                        found = True

                        # Find the index substitution
                        index_map = dict(zip(o_src.external_indices, t_dst.external_indices))
                        for index in {*e_src.external_indices, *e_src.internal_indices}:
                            if index not in index_map:
                                index_map[index] = index.copy(name=f"tmp{idx_counter}")
                                idx_counter += 1

                        def _substitute(tensor: Tensor) -> Base:
                            """Perform the substitution."""
                            if tensor.name == o_src.name:  # noqa: B023
                                return e_src.map_indices(index_map)  # noqa: B023
                            return tensor

                        # Substitute the expression
                        expr[i] = e_dst.apply(_substitute, Tensor)
                        output.pop(j)
                        expr.pop(j)
                        break

            if found:
                break

        # Check if we are done
        if not found:
            if len(output) == 1:
                return expr[0]
            raise ValueError("Complete substitution not possible")


def _tensor_info(tensor: Tensor) -> TensorInfo:
    """Get the information of a tensor (used as a node in a graph)."""
    return (
        tensor.name,
        tuple(i.spin for i in tensor.external_indices),
        tuple(i.space for i in tensor.internal_indices),
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
