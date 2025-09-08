"""Optimisation of expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from albert.opt._gristmill import optimise_gristmill

if TYPE_CHECKING:
    from typing import Any

    from albert.base import Base
    from albert.tensor import Tensor
    from albert.expression import Expression


def optimise(
    exprs: list[Expression],
    method: str = "auto",
    **kwargs: Any,
) -> list[Expression]:
    """Perform common subexpression elimination on the given expression.

    Args:
        outputs: The output tensors for each expression.
        exprs: The expressions to be optimised.
        method: The optimisation method to use. Options are `"auto"`, `"gristmill"`.

    Returns:
        The optimised expressions, as tuples of the output tensor and the expression.
    """
    if method == "gristmill" or method == "auto":
        return optimise_gristmill(exprs, **kwargs)
    else:
        raise ValueError(f"Unknown optimisation method: {method!r}")
