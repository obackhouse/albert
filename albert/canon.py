"""Canonicalisation of expressions."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from albert.base import IMul
from albert.index import Index

if TYPE_CHECKING:
    from typing import Optional

    from albert.base import Base


def canonicalise_indices(expr: Base, extra_indices: Optional[list[Index]] = None) -> Base:
    """Canonicalise the indices of a tensor expression.

    Args:
        expr: The tensor expression to canonicalise.

    Returns:
        The canonicalised tensor expression.
    """
    from albert.tensor import Tensor  # FIXME

    # Get the indices, grouped by spin and space
    index_groups = defaultdict(set)
    for leaf in expr.search_leaves(Tensor):  # type: ignore[type-abstract]
        for index in leaf.external_indices:
            index_groups[(index.spin, index.space)].add(index)
            if index.spin in ("a", "b"):
                index_groups[(index.spin_flip().spin, index.space)].add(index.spin_flip())
    for index in extra_indices or []:
        index_groups[(index.spin, index.space)].add(index)
        if index.spin in ("a", "b"):
            index_groups[(index.spin_flip().spin, index.space)].add(index.spin_flip())
    index_lists = {key: sorted(indices) for key, indices in index_groups.items()}

    # Find the canonical external indices globally
    index_map = {}
    for index in expr.external_indices:
        index_map[index] = index_lists[(index.spin, index.space)].pop(0)

        # If the spin-flipped index exists, remove it to avoid repeat indices with the same name
        if index.spin in ("a", "b"):
            index_flip = index_map[index].spin_flip()
            if index_flip in index_lists.get((index_flip.spin, index_flip.space), []):
                index_lists[(index_flip.spin, index_flip.space)].remove(index_flip)

    def _canonicalise_node(node: IMul) -> Base:
        """Canonicalise a node."""
        index_map_node = index_map.copy()
        index_lists_node = {key: indices.copy() for key, indices in index_lists.items()}

        # Find the canonical internal indices for each term
        for index in node.internal_indices:
            if len(index_lists_node[(index.spin, index.space)]):
                index_map_node[index] = index_lists_node[(index.spin, index.space)].pop(0)
            else:
                # Somehow we ran out of indices? Didn't think this was possible, but we can
                # just make a new one
                index_map_node[index] = Index(
                    f"idx{len(index_map_node)}",
                    space=index.space,
                    spin=index.spin,
                )

            # If the spin-flipped index exists, remove it to avoid repeat indices with the same name
            index_flip = index_map_node[index].spin_flip()
            if index_flip in index_lists_node.get((index_flip.spin, index_flip.space), []):
                index_lists_node[(index_flip.spin, index_flip.space)].remove(index_flip)

        # Canonicalise the node
        return node.map_indices(index_map_node)

    # Find the canonical internal indices for each term
    expr = expr.expand()
    expr = expr.apply(_canonicalise_node, IMul)

    return expr
