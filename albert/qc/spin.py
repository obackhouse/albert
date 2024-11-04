"""Spin integration routines."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING

from albert.algebra import _compose_mul
from albert.index import Index
from albert.scalar import Scalar

if TYPE_CHECKING:
    from albert.base import Base


def ghf_to_uhf(expr: Base, target_rhf: bool = False) -> tuple[Base, ...]:
    """Convert the spin orbital indices of an expression to indices with spin.

    Args:
        expr: The expression to convert.
            target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion
                to UHF is different depending on the target.

    Returns:
        Tuple of expressions resulting from the conversion.
    """
    # Expand the expression into an Add[Mul[Tensor | Scalar]]
    expr = expr.expand()

    # Loop over leaves of the expression tree
    new_exprs: dict[tuple[Index, ...], Base] = defaultdict(Scalar)
    for mul in expr._children:
        # Split the leaves into those that are already unrestricted and those that are not
        uhf_leaves = []
        ghf_leaves = []
        for leaf in mul._children:
            leaf_spins = [index.spin for index in leaf.external_indices]
            if isinstance(leaf, Scalar) or all(spin in ("a", "b") for spin in leaf_spins):
                uhf_leaves.append(leaf)
            else:
                ghf_leaves.append(leaf.as_uhf(target_rhf=target_rhf))

        # Construct the unrestricted leaves
        for ghf_leaves_perm in itertools.product(*ghf_leaves):
            # Check the indices for this permutation have consistent spins
            index_spins = {}
            for index in itertools.chain(*(leaf.external_indices for leaf in ghf_leaves_perm)):
                if index.spin:
                    if index.name not in index_spins:
                        index_spins[index.name] = index.spin
                    elif index_spins[index.name] != index.spin:
                        break
            else:
                # This contribution is valid, add it to the new expression
                new_mul = _compose_mul(*uhf_leaves, *ghf_leaves_perm)
                new_exprs[new_mul.external_indices] += new_mul

    return tuple(new_exprs.values())


def uhf_to_rhf(expr: Base) -> Base:
    """Convert the indices with spin of an expression to restricted indices.

    Args:
        expr: The expression to convert.

    Returns:
        Tuple of expressions resulting from the conversion.
    """
    # Expand the expression into an Add[Mul[Tensor | Scalar]]
    expr = expr.expand()

    # Loop over leaves of the expression tree
    new_expr: Base = Scalar(0)
    for mul in expr._children:
        leaves: list[Base] = []
        for leaf in mul._children:
            leaf_spins = [index.spin for index in leaf.external_indices]
            if isinstance(leaf, Scalar) or all(spin == "r" for spin in leaf_spins):
                leaves.append(leaf)
            else:
                leaves.append(leaf.as_rhf())

        # Construct the new expression
        new_expr += _compose_mul(*leaves)

    return new_expr


def ghf_to_rhf(expr: Base) -> Base:
    """Convert the indices with spin of an expression to restricted indices.

    Args:
        expr: The expression to convert.

    Returns:
        Tuple of expressions resulting from the conversion.
    """
    # Convert to UHF
    uhf_exprs = ghf_to_uhf(expr, target_rhf=True)

    # Partially canonicalise
    canon_exprs: dict[tuple[Index, ...], Base] = defaultdict(Scalar)
    for expr in uhf_exprs:
        expr = expr.canonicalise()
        canon_exprs[expr.external_indices] += expr
    uhf_exprs = canon_exprs

    # Convert to RHF
    rhf_exprs = sum((uhf_to_rhf(expr) for expr in uhf_exprs.values()), Scalar(0))

    return rhf_exprs
