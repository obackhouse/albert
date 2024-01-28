"""Interface to `pdaggerq`.
"""

import re
from numbers import Number

from einfun.algebra import Mul, Add
from einfun.qc.ghf import Fock, ERI, T1, T2, T3


def import_from_pdaggerq(terms):
    """Import terms from `pdaggerq` output into the internal format.

    Parameters
    ----------
    terms : list of list of str
        Terms from `pdaggerq` output, corresponding to a single tensor
        output.

    Returns
    -------
    expr : Algebraic
        Expression in the internal format.
    """

    contractions = []
    for term in terms:
        # Convert symbols
        symbols = [_convert_symbol(symbol) for symbol in term]

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


class PermutationOperator:
    """Permutation operator.
    """

    def __init__(self, i, j):
        self.indices = (i, j)


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


def _convert_symbol(symbol):
    """Convert a `pdaggerq` symbol to the internal format.

    Parameters
    ----------
    symbol : str
        Symbol to convert.

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
        symbols = tuple(symbol[2:-1].split(","))
        return Fock[symbols]

    elif re.match(r"<[a-z],[a-z]\|\|[a-z],[a-z]>", symbol):
        # <i,j||k,l>
        symbols = tuple(symbol[1:-1].replace("||", ",").split(","))
        return ERI[symbols]

    elif re.match(r"t1\([a-z],[a-z]\)", symbol):
        # t1(i,j)
        symbols = tuple(symbol[3:-1].split(","))
        symbols = (symbols[1], symbols[0])
        return T1[symbols]

    elif re.match(r"t2\([a-z],[a-z],[a-z],[a-z]\)", symbol):
        # t2(i,j,k,l)
        symbols = tuple(symbol[3:-1].split(","))
        symbols = (symbols[2], symbols[3], symbols[0], symbols[1])
        return T2[symbols]

    elif re.match(r"t3\([a-z],[a-z],[a-z],[a-z],[a-z],[a-z]\)", symbol):
        # t3(i,j,k,l,m,n)
        symbols = tuple(symbol[3:-1].split(","))
        symbols = (symbols[3], symbols[4], symbols[5], symbols[0], symbols[1], symbols[2])
        return T3[symbols]

    elif re.match(r"P\([a-z],[a-z]\)", symbol):
        # P(i,j)
        symbols = tuple(symbol[2:-1].split(","))
        return PermutationOperator(*symbols)

    else:
        raise ValueError(f"Unknown symbol {symbol}")
