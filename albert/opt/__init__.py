"""Optimisation of expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from albert.opt._gristmill import optimise_gristmill
from albert.opt.cse import optimise as optimise_albert
from albert.opt._brute import eliminate_and_factorise_common_subexpressions

if TYPE_CHECKING:
    from typing import Any, Literal
    from albert.expression import Expression


def optimise(
    exprs: list[Expression],
    method: Literal["auto", "gristmill", "albert", "legacy"] = "auto",
    **kwargs: Any,
) -> list[Expression]:
    """Perform common subexpression elimination on the given expression.

    Args:
        exprs: The expressions to be optimised.
        method: The optimisation method to use.
        **kwargs: Additional keyword arguments to pass to the optimisation method.

    Returns:
        The optimised expressions, as tuples of the output tensor and the expression.
    """
    if method == "auto":
        try:
            return optimise_gristmill(exprs, **kwargs)
        except ImportError:
            return optimise_albert(exprs, **kwargs)
    elif method == "gristmill":
        return optimise_gristmill(exprs, **kwargs)
    elif method == "albert":
        return optimise_albert(exprs, **kwargs)
    elif method == "legacy":
        return sum(
            [eliminate_and_factorise_common_subexpressions(expr, **kwargs) for expr in exprs],
            [],
        )
    else:
        raise ValueError(f"Unknown optimisation method: {method!r}")
