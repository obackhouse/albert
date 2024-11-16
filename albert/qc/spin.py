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


def ghf_to_uhf(
    expr: Base,
    target_rhf: bool = False,
    canonicalise: bool = True,
) -> tuple[Base, ...]:
    """Convert the spin orbital indices of an expression to indices with spin.

    Args:
        expr: The expression to convert.
        target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion to UHF
            is different depending on the target.
        canonicalise: Whether to canonicalise the tensors after converting them to UHF basis.

    Returns:
        Tuple of expressions resulting from the conversion.
    """
    # Expand the expression into an Add[Mul[Tensor | Scalar]]
    expr = expr.expand()

    # Loop over leaves of the expression tree
    new_exprs: dict[tuple[Index, ...], Base] = defaultdict(Scalar)
    for mul in expr._children:
        # Split the leaves into those that are already unrestricted and those that are not
        scalars = []
        tensors = []
        for leaf in mul._children:
            if isinstance(leaf, Scalar):
                scalars.append(leaf)
            else:
                leaf_as_uhf = leaf.as_uhf(target_rhf=target_rhf)
                if canonicalise:
                    leaf_as_uhf = tuple(e.canonicalise() for e in leaf_as_uhf)
                tensors.append(leaf_as_uhf)

        # Construct the unrestricted leaves
        for tensors_perm in itertools.product(*tensors):
            # Check the indices for this permutation have consistent spins
            index_spins = {}
            for index in itertools.chain(*(leaf.external_indices for leaf in tensors_perm)):
                if index.spin:
                    if index.name not in index_spins:
                        index_spins[index.name] = index.spin
                    elif index_spins[index.name] != index.spin:
                        break
            else:
                # This contribution is valid, add it to the new expression
                new_mul = _compose_mul(*scalars, *tensors_perm)
                new_exprs[new_mul.external_indices] += new_mul

    # Expand and collect
    new_exprs = tuple(expr.expand().collect() for expr in new_exprs.values())

    return new_exprs


def uhf_to_rhf(expr: Base, canonicalise: bool = True) -> Base:
    """Convert the indices with spin of an expression to restricted indices.

    Args:
        expr: The expression to convert.
        canonicalise: Whether to canonicalise the tensors after converting them to RHF basis.

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
            if isinstance(leaf, Scalar):
                leaves.append(leaf)
            else:
                leaf_as_rhf = leaf.as_rhf()
                if canonicalise:
                    leaf_as_rhf = leaf_as_rhf.canonicalise()
                leaves.append(leaf_as_rhf)

        # Construct the new expression
        new_mul = _compose_mul(*leaves)
        new_expr += new_mul

    # Expand and collect
    new_expr = new_expr.expand().collect()

    return new_expr


def ghf_to_rhf(expr: Base, canonicalise: bool = True) -> Base:
    """Convert the indices with spin of an expression to restricted indices.

    Args:
        expr: The expression to convert.
        canonicalise: Whether to canonicalise the tensors after converting them between bases.

    Returns:
        Tuple of expressions resulting from the conversion.
    """
    # Convert to UHF
    uhf_exprs = ghf_to_uhf(expr, target_rhf=True, canonicalise=canonicalise)

    # Convert to RHF
    rhf_exprs = sum((uhf_to_rhf(expr, canonicalise=canonicalise) for expr in uhf_exprs), Scalar(0))

    return rhf_exprs
