"""Interface to `pdaggerq`.
"""

import re
from numbers import Number

from albert.algebra import Add, Mul
from albert.qc.ghf import ERI, L1, L2, L3, T1, T2, T3, Fock
from albert.qc.uhf import SpinIndex


def import_from_pdaggerq(terms, index_spins=None):
    """Import terms from `pdaggerq` output into the internal format.

    Parameters
    ----------
    terms : list of list of str
        Terms from `pdaggerq` output, corresponding to a single tensor
        output.
    index_spins : dict, optional
        Dictionary mapping indices to spins. If not provided, the
        indices are not constrained to a single spin. Default value is
        `None`.

    Returns
    -------
    expr : Algebraic
        Expression in the internal format.
    """

    # Get the index spins
    if index_spins is None:
        index_spins = {}

    contractions = []
    for term in terms:
        # Convert symbols
        symbols = [_convert_symbol(symbol, index_spins=index_spins) for symbol in term]

        # Remove the permutation operators
        perm_operators = [symbol for symbol in symbols if isinstance(symbol, PermutationOperator)]
        symbols = [symbol for symbol in symbols if not isinstance(symbol, PermutationOperator)]
        part = Mul(*symbols)
        for perm_operator in perm_operators:
            index_map = {
                perm_operator.indices[0]: perm_operator.indices[1],
                perm_operator.indices[1]: perm_operator.indices[0],
            }
            part = part - part.map_indices(index_map)

        # Add to the list of contractions
        contractions.append(part)

    # Add all contractions together
    expr = Add(*contractions)

    return expr


class PermutationOperatorSymbol:
    """Permutation operator symbol."""

    def __getitem__(self, indices):
        """Create a permutation operator."""
        return PermutationOperator(*indices)


class PermutationOperator:
    """Permutation operator."""

    def __init__(self, i, j):
        self.indices = (i, j)


P = PermutationOperatorSymbol()


def _is_number(symbol):
    """Check if something is a number.

    Parameters
    ----------
    symbol : str
        Symbol to check.

    Returns
    -------
    out : bool
        True if `symbol` is a number, False otherwise.
    """

    if isinstance(symbol, Number):
        return True
    else:
        try:
            float(symbol)
            return True
        except ValueError:
            return False


def _convert_symbol(symbol, index_spins=None):
    """Convert a `pdaggerq` symbol to the internal format.

    Parameters
    ----------
    symbol : str
        Symbol to convert.
    index_spins : dict, optional
        Dictionary mapping indices to spins. If not provided, the
        indices are not constrained to a single spin. Default value is
        `None`.

    Returns
    -------
    out : Tensor or Number
        Converted symbol.
    """

    if _is_number(symbol):
        # factor
        return float(symbol)

    elif re.match(r"f\([a-z],[a-z]\)", symbol):
        # f(i,j)
        indices = tuple(symbol[2:-1].split(","))
        tensor_symbol = Fock

    elif re.match(r"<[a-z],[a-z]\|\|[a-z],[a-z]>", symbol):
        # <i,j||k,l>
        indices = tuple(symbol[1:-1].replace("||", ",").split(","))
        tensor_symbol = ERI

    elif re.match(r"t1\([a-z],[a-z]\)", symbol):
        # t1(i,j)
        indices = tuple(symbol[3:-1].split(","))
        indices = (indices[1], indices[0])
        tensor_symbol = T1

    elif re.match(r"t2\([a-z],[a-z],[a-z],[a-z]\)", symbol):
        # t2(i,j,k,l)
        indices = tuple(symbol[3:-1].split(","))
        indices = (indices[2], indices[3], indices[0], indices[1])
        tensor_symbol = T2

    elif re.match(r"t3\([a-z],[a-z],[a-z],[a-z],[a-z],[a-z]\)", symbol):
        # t3(i,j,k,l,m,n)
        indices = tuple(symbol[3:-1].split(","))
        indices = (indices[3], indices[4], indices[5], indices[0], indices[1], indices[2])
        tensor_symbol = T3

    elif re.match(r"l1\([a-z],[a-z]\)", symbol):
        # t1(i,j)
        indices = tuple(symbol[3:-1].split(","))
        indices = (indices[1], indices[0])
        tensor_symbol = L1

    elif re.match(r"l2\([a-z],[a-z],[a-z],[a-z]\)", symbol):
        # t2(i,j,k,l)
        indices = tuple(symbol[3:-1].split(","))
        indices = (indices[2], indices[3], indices[0], indices[1])
        tensor_symbol = L2

    elif re.match(r"l3\([a-z],[a-z],[a-z],[a-z],[a-z],[a-z]\)", symbol):
        # t3(i,j,k,l,m,n)
        indices = tuple(symbol[3:-1].split(","))
        indices = (indices[3], indices[4], indices[5], indices[0], indices[1], indices[2])
        tensor_symbol = L3

    elif re.match(r"P\([a-z],[a-z]\)", symbol):
        # P(i,j)
        indices = tuple(symbol[2:-1].split(","))
        tensor_symbol = P

    else:
        raise ValueError(f"Unknown symbol {symbol}")

    # Convert the indices to SpinIndex
    indices = tuple(
        SpinIndex(index, index_spins[index]) if index in index_spins else index for index in indices
    )

    return tensor_symbol[indices]
