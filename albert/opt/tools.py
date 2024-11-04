"""Tools for optimisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from albert.tensor import Tensor

if TYPE_CHECKING:
    from albert.base import Base


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
