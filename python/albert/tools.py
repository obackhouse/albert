"""Tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from albert._core import CostEstimate, Tensor, estimate_cost, scaling_exponent
from albert.options import ScalingLimits, ScalingMetric

if TYPE_CHECKING:
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


def print_cost_report(
    expressions: list[Expression],
    *,
    metadata: Metadata,
    cost_model: CostModel | None = None,
    scaling_limits: ScalingLimits | None = None,
    title: str | None = None,
    end: str = "\n",
) -> None:
    """Print a cost report for a list of expressions.

    Args:
        expressions: List of expressions to analyze.
        metadata: The tensor metadata.
        cost_model: The cost model for estimating costs. Note that :attr:`CostModel.apply_cost_factors`
            is honoured and should be suppressed if not desired. Set to ``None`` if no cost model is
            provided.
        scaling_limits: The scaling limits used for the CSE. This is used to determine which spaces
            to report scaling for. Set to ``None`` if no scaling limits are provided.
        title: Optional title to print before the report.
        end: String to print at the end of the report.
    """
    if not cost_model and not scaling_limits:
        return
    total_cost, max_scaling = cost_report(
        expressions, metadata=metadata, cost_model=cost_model, scaling_limits=scaling_limits
    )
    full_title = f"Cost report {'(' if title else ''}{title if title else ''}{')' if title else ''}"
    print(full_title.center(28))
    print("-" * 28)

    if total_cost is not None:
        print("FLOPs:")
        print(f"- total:            {total_cost.total:.2e}")
        print(f"- multiplications:  {total_cost.multiplications:.2e}")
        print(f"- additions:        {total_cost.additions:.2e}")
        print(f"- reads:            {total_cost.reads:.2e}")
        print(f"- writes:           {total_cost.writes:.2e}")

    if max_scaling is not None:
        for which in ("time", "memory"):
            if getattr(max_scaling, which):
                print(f"Scaling ({which}):")
            for space, exponent in getattr(max_scaling, which).items():
                space_str = ",".join(space) if isinstance(space, tuple) else space
                print(f"- {space_str}:".ljust(19), f"{exponent}")

    print("-" * 28, end=end)


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
