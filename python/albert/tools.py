"""Tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from albert._core import CostEstimate, Node, Tensor, estimate_cost, scaling_exponent, topological_sort
from albert.options import ScalingLimits, ScalingMetric

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import Any

    from albert._core import Expression, Node
    from albert.metadata import Metadata
    from albert.options import CostModel


@overload
def cost_report(
    expressions: list[Expression],
    *,
    metadata: Metadata,
    cost_model: None = None,
    scaling_limits: None = None,
) -> tuple[None, None]: ...
@overload
def cost_report(
    expressions: list[Expression],
    *,
    metadata: Metadata,
    cost_model: None = None,
    scaling_limits: ScalingLimits | None = None,
) -> tuple[None, ScalingLimits | None]: ...
@overload
def cost_report(
    expressions: list[Expression],
    *,
    metadata: Metadata,
    cost_model: CostModel | None = None,
    scaling_limits: None = None,
) -> tuple[CostEstimate | None, None]: ...
@overload
def cost_report(
    expressions: list[Expression],
    *,
    metadata: Metadata,
    cost_model: CostModel,
    scaling_limits: ScalingLimits,
) -> tuple[CostEstimate, ScalingLimits]: ...


def cost_report(
    expressions: list[Expression],
    *,
    metadata: Metadata,
    cost_model: CostModel | None = None,
    scaling_limits: ScalingLimits | None = None,
) -> tuple[CostEstimate | None, ScalingLimits | None]:
    """Get a cost report for a list of expressions.

    Args:
        expressions: List of expressions to analyze.
        metadata: The tensor metadata.
        cost_model: The cost model for estimating costs. Note that :attr:`CostModel.apply_cost_factors`
            is honoured and should be suppressed if not desired. Set to ``None`` if no cost model is
            provided.
        scaling_limits: The scaling limits used for the CSE. This is used to determine which spaces
            to report scaling for. Set to ``None`` if no scaling limits are provided.

    Returns:
        The total cost estimate and the observed maximum scaling limits.
    """
    if cost_model is not None:
        costs = [estimate_cost(expr.rhs, metadata=metadata, cost_model=cost_model) for expr in expressions]
        total_cost = CostEstimate(
            **{
                name: sum(getattr(cost, name) for cost in costs)
                for name in ("multiplications", "additions", "reads", "writes", "total")
            }
        )
    else:
        total_cost = None

    if scaling_limits is not None:
        max_scaling = ScalingLimits(
            **{
                which: {
                    space: max(
                        scaling_exponent(expr.rhs, space, ScalingMetric(which), metadata) for expr in expressions
                    )
                    for space in getattr(scaling_limits, which)
                }
                for which in ("time", "memory")
            }
        )
    else:
        max_scaling = None

    return total_cost, max_scaling


def cost_report_json(
    expressions: list[Expression],
    *,
    metadata: Metadata,
    cost_model: CostModel | None = None,
    scaling_limits: ScalingLimits | None = None,
) -> dict[str, Any]:
    """Get a cost report for a list of expressions as a JSON-serializable dictionary.

    Args:
        expressions: List of expressions to analyze.
        metadata: The tensor metadata.
        cost_model: The cost model for estimating costs. Note that :attr:`CostModel.apply_cost_factors`
            is honoured and should be suppressed if not desired. Set to ``None`` if no cost model is
            provided.
        scaling_limits: The scaling limits used for the CSE. This is used to determine which spaces
            to report scaling for. Set to ``None`` if no scaling limits are provided.

    Returns:
        A dictionary containing the total cost estimate and the observed maximum scaling limits.
    """
    total_cost, max_scaling = cost_report(
        expressions, metadata=metadata, cost_model=cost_model, scaling_limits=scaling_limits
    )
    report: dict[str, Any] = {}
    if total_cost is not None:
        report["cost"] = {
            "total": total_cost.total,
            "multiplications": total_cost.multiplications,
            "additions": total_cost.additions,
            "reads": total_cost.reads,
            "writes": total_cost.writes,
        }
    if max_scaling is not None:
        report["scaling"] = {
            which: {space: exponent for space, exponent in getattr(max_scaling, which).items()}
            for which in ("time", "memory")
        }
    return report


def split_by_dependencies(
    expressions: list[Expression],
    dependency_symbols: Collection[str],
    *,
    metadata: Metadata,
) -> tuple[list[Expression], list[Expression]]:
    """Split equations into independent and dependent parts by tensor dependencies.

    Any equation whose RHS uses a tensor with a symbol in ``dependency_symbols``
    is dependent. This classification propagates through intermediate definitions:
    an equation that uses a dependent intermediate is also dependent. Equations
    whose LHS symbol is in ``dependency_symbols`` are iterative as well.

    Tensor definitions are matched by both symbol and spaces, so distinct arrays
    represented by the same symbol remain independent. Both returned lists are in
    dependency order.

    Args:
        expressions: Equations to split.
        dependency_symbols: Tensor symbols that identify iterative dependencies,
        whose dependencies should mark equations as dependent.
        metadata: Metadata used to resolve tensor indices to spaces.

    Returns:
        A pair ``(independent, dependent)`` of equations, each in dependency order.
    """
    if isinstance(dependency_symbols, str):
        raise TypeError("dependency_symbols must be a collection of tensor symbols, not a string")
    symbols = set(dependency_symbols)
    if any(not isinstance(symbol, str) for symbol in symbols):
        raise TypeError("dependency_symbols must contain only strings")

    index_spaces = {index: space for space, indices in metadata.indices.items() for index in indices}

    def tensor_key(tensor: Tensor) -> tuple[str, tuple[str, ...]]:
        try:
            spaces = tuple(index_spaces[index] for index in tensor.indices)
        except KeyError as error:
            raise ValueError(f"index {error.args[0]!r} does not belong to a metadata space") from error
        return tensor.symbol, spaces

    def tensor_nodes(node: Node) -> list[Tensor]:
        if isinstance(node, Tensor):
            return [node]
        tensors: list[Tensor] = []
        for child in node.children:
            tensors.extend(tensor_nodes(child))
        return tensors

    ordered = topological_sort(expressions, metadata)
    dependent_definitions: set[tuple[str, tuple[str, ...]]] = set()
    independent: list[Expression] = []
    dependent: list[Expression] = []
    for expression in ordered:
        rhs_tensors = tensor_nodes(expression.rhs)
        is_dependent = expression.lhs.symbol in symbols or any(
            tensor.symbol in symbols or tensor_key(tensor) in dependent_definitions for tensor in rhs_tensors
        )
        if is_dependent:
            dependent.append(expression)
            dependent_definitions.add(tensor_key(expression.lhs))
        else:
            independent.append(expression)
    return independent, dependent


def next_unused_index(node: Node, space: str, *, metadata: Metadata, skip: int = 0) -> str:
    """Get the next unused index in a given space for a node.

    Args:
        node: The node to check for used indices.
        space: The space to check for unused indices.
        metadata: The tensor metadata.
        skip: The number of unused indices to skip before returning one.

    Returns:
        The next unused index in the given space.
    """
    used_indices = set()
    for tensor in node.search(lambda n: isinstance(n, Tensor)):
        used_indices.update(tensor.external_indices)
    for index in metadata.indices[space]:
        if index not in used_indices:
            if skip == 0:
                return index
            skip -= 1
    raise ValueError(f"no available indices in space {space}")
