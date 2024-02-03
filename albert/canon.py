"""Canonicalisation of terms and expressions.
"""

from albert.algebra import Add


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

    # Find the canonical external indices globally
    index_map = {}
    for index in expr.external_indices:
        for indices, index_set in zip(index_lists, index_sets):
            if index in index_set:
                index_map[index] = indices.pop(0)
                break
        else:
            raise ValueError(f"Index {index} not found in any index group")

    # Find the canonical dummy indices for each term in the addition
    for i, arg in enumerate(args):
        index_map_i = index_map.copy()
        index_lists_i = [indices.copy() for indices in index_lists]
        for index in arg.dummy_indices:
            for indices, index_set in zip(index_lists_i, index_sets):
                if index in index_set:
                    index_map_i[index] = indices.pop(0)
                    break
            else:
                raise ValueError(f"Index {index} not found in any index group")

        # Replace the indices in the term
        args[i] = arg.map_indices(index_map_i)

    # Build the canonicalised expression
    expr = Add(*args)

    return expr
