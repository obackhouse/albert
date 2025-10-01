"""Canonicalisation of expressions."""

from __future__ import annotations

import functools
import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

from albert.algebra import Algebraic, Mul
from albert.index import Index
from albert.scalar import Scalar
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Callable, Generator, Optional

    from albert.base import Base


T = TypeVar("T", contravariant=True)


class SupportsDunderLT(Protocol[T]):
    """Protocol for objects that support the less than comparison."""

    def __lt__(self, __other: T) -> bool:
        """Less than comparison for canonicalisation."""
        pass


def iter_equivalent_forms(expr: Base) -> Generator[Base, None, None]:
    """Iterate over all equivalent forms of a tensor expression.

    Args:
        expr: The tensor expression to iterate over.

    Yields:
        All equivalent forms of the tensor expression.
    """

    @functools.lru_cache(maxsize=None)
    def _symmetry(node: Tensor) -> list[Base]:
        if node.symmetry is None:
            return [node]
        return list(node.symmetry(node))

    def _iter_variants(node: Base) -> Generator[Base, None, None]:
        if isinstance(node, Scalar):
            yield node
            return
        if isinstance(node, Tensor):
            yield from _symmetry(node)
            return
        assert node.children is not None
        variants = [list(_iter_variants(child)) if child else [] for child in node.children]
        for combo in itertools.product(*variants):
            yield node.copy(*combo)

    yield from _iter_variants(expr)


def canonicalise_exhaustive(
    expr: Base,
    key: Callable[[Base], SupportsDunderLT[Any]] | None = None,
) -> Base:
    """Canonicalise a tensor expression exhaustively.

    Args:
        expr: The tensor expression to canonicalise.
        key: A key function to use for canonicalisation. If ``None``, a default key function is
            used that sorts algebraic sums by their children, ignoring scalar factors.

    Returns:
        The canonicalised tensor expression.
    """

    def _iter_equivalent_forms(expr: Base) -> Generator[Base, None, None]:
        """Iterate over all equivalent forms of a tensor expression."""
        internal_indices = sorted(expr.internal_indices)
        categories = [index.category for index in internal_indices]
        for perm in itertools.permutations(range(len(internal_indices))):
            if all(categories[i] == categories[j] for i, j in enumerate(perm)):
                index_map = dict(zip(internal_indices, (internal_indices[i] for i in perm)))
                yield expr.map_indices(index_map).canonicalise()

    if key is None:

        def key(e: Base) -> SupportsDunderLT[Any]:
            """Key function for canonicalisation."""
            if isinstance(e, Algebraic):
                return tuple(sorted(filter(lambda x: not isinstance(x, Scalar), e.children or [])))
            return e

    expr = min(_iter_equivalent_forms(expr), key=key)

    return expr


def canonicalise_indices(
    expr: Base,
    extra_indices: Optional[list[Index]] = None,
    which: Literal["all", "external", "internal"] = "all",
) -> Base:
    """Canonicalise the indices of a tensor expression.

    Args:
        expr: The tensor expression to canonicalise.
        extra_indices: Extra indices to consider when canonicalising.
        which: Which indices to canonicalise.

    Returns:
        The canonicalised tensor expression.
    """
    # Get the indices, grouped by category
    index_groups = defaultdict(set)
    for leaf in expr.search(Tensor):
        for index in leaf.external_indices:
            index_groups[index.category].add(index)
            index_flip = index.spin_flip()
            if index != index_flip:
                index_groups[index_flip.category].add(index_flip)
    for index in extra_indices or []:
        index_groups[index.category].add(index)
        index_flip = index.spin_flip()
        if index != index_flip:
            index_groups[index_flip.category].add(index_flip)
    index_lists = {key: sorted(indices) for key, indices in index_groups.items()}

    # Find the canonical external indices globally
    index_map = {}
    for index in expr.external_indices:
        index_map[index] = index_lists[index.category].pop(0)

        # If the spin-flipped index exists, remove it to avoid repeat indices with the same name
        index_flip = index_map[index].spin_flip()
        if index_map[index] != index_flip:
            if index_flip in index_lists.get(index_flip.category, []):
                index_lists[index_flip.category].remove(index_flip)

    def _canonicalise_node(node: Mul) -> Base:
        """Canonicalise a node."""
        index_map_node = index_map.copy()
        index_lists_node = {key: indices.copy() for key, indices in index_lists.items()}

        # Find the canonical internal indices for each term
        for index in node.internal_indices:
            if index_lists_node[index.category]:
                index_map_node[index] = index_lists_node[index.category].pop(0)
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
            if index_flip in index_lists_node.get(index_flip.category, []):
                index_lists_node[index_flip.category].remove(index_flip)

        # Remove mappings for indices we don't want to change
        for src, dst in list(index_map.items()):
            touches_internal = src in node.internal_indices or dst in node.internal_indices
            touches_external = src in node.external_indices or dst in node.external_indices
            if which == "external" and touches_internal:
                del index_map_node[src]
            elif which == "internal" and touches_external:
                del index_map_node[src]

        # Canonicalise the node
        return node.map_indices(index_map_node)

    # Find the canonical internal indices for each term
    expr = expr.expand()
    expr = expr.apply(_canonicalise_node, Mul)

    return expr
