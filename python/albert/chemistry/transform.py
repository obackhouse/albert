"""Node filters for chemistry-specific algebra transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from albert._core import Add, Tensor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from albert._core import Node


def expand_antisymmetrized_eris(
    node: Node,
    *,
    transform_like: Mapping[str, str] | None = None,
) -> Node:
    """Expand an antisymmetrized ERI in terms of ordinary ERIs.

    In physicists' notation, ``v_a[i, j, k, l]`` is
    ``v[i, j, k, l] - v[i, j, l, k]``. Keeping this as a node filter makes
    the expansion available to other chemistry passes as well as
    :func:`albert.chemistry.spin.spin_integrate`.
    """
    if not isinstance(node, Tensor):
        return node
    symbol = node.symbol if transform_like is None else transform_like.get(node.symbol, node.symbol)
    if symbol != "v_a":
        return node
    if node.rank != 4:
        raise ValueError(f"antisymmetrized ERI tensor {node.symbol!r} must have rank 4")
    i, j, k, ell = node.indices
    direct = Tensor("v", [i, j, k, ell])
    exchange = Tensor("v", [i, j, ell, k])
    return Add([direct, -exchange])
