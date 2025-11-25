"""Functionality specific to quantum chemistry applications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from albert.qc._pdaggerq import import_from_pdaggerq
from albert.qc._wick import import_from_wick
from albert.qc.spin import ghf_to_rhf, ghf_to_uhf
from albert.tensor import Tensor
from albert.expression import Expression

if TYPE_CHECKING:
    from typing import Any, Iterable, Literal


def import_expression(
    external: Any,
    package: Literal["pdaggerq", "wick"] = "pdaggerq",
    index_order: Iterable[str] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> Expression:
    """Import an expression from a third-party quantum chemistry package.

    Args:
        external: The external expression to import. The exact format is specified by the
            individual importers.
        package: The code from which to import the expression.
        index_order: The desired order of external index labels in the imported expression. The
            indices of the left-hand side of the expression will be sorted such that their labels
            match this order.
        name: The name to assign to the LHS tensor.
        **kwargs: Additional keyword arguments to pass to the importer.

    Returns:
        The imported expression.
    """
    # Import the RHS
    if package == "pdaggerq":
        rhs = import_from_pdaggerq(external, **kwargs)
    elif package == "wick":
        rhs = import_from_wick(external, **kwargs)
    else:
        raise ValueError(f"Unknown package: {package}")
    rhs = rhs.canonicalise(indices=True)

    # Get the LHS
    if index_order is None:
        indices = rhs.external_indices
    else:
        indices = tuple(sorted(rhs.external_indices, key=lambda i: list(index_order).index(i.name)))
    lhs = Tensor(*indices, name=name)

    return Expression(lhs, rhs)


def adapt_spin(
    expr: Expression | Iterable[Expression],
    target_spin: Literal["rhf", "uhf"],
) -> tuple[Expression, ...]:
    """Adapt the spin representation of a quantum chemistry expression.

    Args:
        expr: The expression(s) to adapt.
        target_spin: The target spin representation.

    Returns:
        The adapted expressions. For `"rhf"`, this is a tuple with a single expression. For `"uhf"`,
        this is a tuple with one expression per spin case.
    """
    if isinstance(expr, Expression):
        expr = (expr,)

    # Convert the RHS
    if target_spin == "rhf":
        rhs_list = [ghf_to_rhf(e.rhs) for e in expr]
        lhs_list = [e.lhs for e in expr]
    elif target_spin == "uhf":
        rhs_list = []
        lhs_list = []
        for e in expr:
            rhs_parts = ghf_to_uhf(e.rhs)
            rhs_list.extend(rhs_parts)
            lhs_list.extend([e.lhs for _ in rhs_parts])
    else:
        raise ValueError(f"Unknown target spin: {target_spin}")

    # Get the LHS for each case
    exprs = []
    for lhs, rhs in zip(lhs_list, rhs_list):
        spins = {index.name: index.spin for index in rhs.external_indices}
        index_map = {index: index.copy(spin=spins[index.name]) for index in lhs.external_indices}
        exprs.append(Expression(lhs.map_indices(index_map), rhs))

    return tuple(exprs)
