"""Canonicalisation of terms and expressions.
"""

from albert.algebra import Add
from albert.qc.uhf import SpinIndex


def canonicalise_indices(expr, *index_groups):
    """Canonicalise the indices of a tensor expression.

    Parameters
    ----------
    expr : Algebraic
        The tensor expression to canonicalise.
    index_groups : tuple of tuple of str
        Groups of equivalent indices, used to replace indices throughout
        the expression in a canonical fashion.

    Returns
    -------
    expr : Algebraic
        The canonicalised tensor expression.
    """

    # Expand parentheses
    expr = expr.expand()

    # Canonicalise the expression
    expr = expr.canonicalise()

    # Get the index lists and sets
    index_sets = [set(indices) for indices in index_groups]
    index_lists = [sorted(indices) for indices in index_groups]

    # Get the arguments as a list of Mul objects
    args = list(expr.args) if isinstance(expr, Add) else [expr]

    # Spin-aware helper functions
    def _unpack_index(index):
        if isinstance(index, SpinIndex):
            return index.index, index.spin
        return index, None

    def _set_index(index_map, index, new_index, spin):
        if spin is None:
            index_map[index] = new_index
        else:
            index_map[index] = SpinIndex(new_index, spin)
            index_map[index.spin_flip()] = index_map[index].spin_flip()

    # Find the canonical external indices globally
    index_map = {}
    for external_index in expr.external_indices:
        index, spin = _unpack_index(external_index)

        # Find the group and set the canonical index
        for indices, index_set in zip(index_lists, index_sets):
            if index in index_set:
                new_index = indices.pop(0)
                _set_index(index_map, external_index, new_index, spin)
                break
        else:
            raise ValueError(f"Index {index} not found in any index group")

    # Find the canonical dummy indices for each term in the addition
    for i, arg in enumerate(args):
        index_map_i = index_map.copy()
        index_lists_i = [indices.copy() for indices in index_lists]

        for dummy_index in arg.dummy_indices:
            index, spin = _unpack_index(dummy_index)

            # Find the group and set the canonical index
            for indices, index_set in zip(index_lists_i, index_sets):
                if index in index_set:
                    new_index = indices.pop(0)
                    _set_index(index_map_i, dummy_index, new_index, spin)
                    break
            else:
                raise ValueError(f"Index {index} not found in any index group")

        # Replace the indices in the term
        args[i] = arg.map_indices(index_map_i)

    # Build the canonicalised expression
    expr = Add(*args)

    return expr
