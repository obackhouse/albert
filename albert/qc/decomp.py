"""Decomposition routines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from albert.index import Index
from albert.qc.rhf import CDERI as RCDERI
from albert.qc.rhf import ERI as RERI
from albert.qc.uhf import CDERI as UCDERI
from albert.qc.uhf import ERI as UERI
from albert.tensor import Tensor

if TYPE_CHECKING:
    from albert.base import Base


def density_fit(expr: Base) -> Base:
    """Apply density fitting to an expression.

    Swaps all `ERI` tensors for products of `CDERI` tensors.

    Args:
        expr: The expression to apply density fitting to.

    Returns:
        The density fitted expression.
    """
    memo = dict(counter=0)

    def _density_fit(tensor: Tensor) -> Base:
        """Apply density fitting to a tensor."""
        if not isinstance(tensor, (RERI, UERI)):
            return tensor
        index = Index(f"x{memo['counter']}", space="x")
        memo["counter"] += 1
        cls = RCDERI if isinstance(tensor, RERI) else UCDERI
        bra = cls(index, *tensor.external_indices[:2])
        ket = cls(index, *tensor.external_indices[2:])
        return bra * ket

    return expr.apply(_density_fit, Tensor)
