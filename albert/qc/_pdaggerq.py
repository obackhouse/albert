"""Interface to `pdaggerq`.
"""

import re
from numbers import Number

from albert.algebra import Add, Mul
from albert.qc.ghf import ERI, L1, L2, L3, T1, T2, T3, R1ip, R2ip, R3ip, R1ea, R2ea, R3ea, R1ee, R2ee, R3ee, Delta, Fock, SingleERI
from albert.qc.index import Index


def import_from_pdaggerq(terms, index_spins=None, index_spaces=None):
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
    index_spaces : dict, optional
        Dictionary mapping indices to spaces. If not provided, the
        indices are assigned to the spaces based on their names. Default
        value is `None`.

    Returns
    -------
    expr : Algebraic
        Expression in the internal format.
    """

    # Get the index spins and spaces
    if index_spins is None:
        index_spins = {}
    if index_spaces is None:
        index_spaces = {}

    contractions = []
    for term in terms:
        # Convert symbols
        symbols = [
            _convert_symbol(symbol, index_spins=index_spins, index_spaces=index_spaces)
            for symbol in term
        ]

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


def _to_space(index, which="full"):
    """Convert an index to a space.

    Parameters
    ----------
    index : str
        Index to convert.
    which : str, optional
        Which space to convert to. Default value is `full`.

    Returns
    -------
    out : str
        Space.
    """

    if index in "ijklmnot" or index.startswith("o"):
        if which == "full":
            return "o"
        elif which == "active":
            return "O"
        elif which == "inactive":
            return "i"
    elif index in "abcdefgh" or index.startswith("v"):
        if which == "full":
            return "v"
        elif which == "active":
            return "V"
        elif which == "inactive":
            return "a"

    raise ValueError(f"Unknown index {index}")


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


def _convert_symbol(symbol, index_spins=None, index_spaces=None):
    """Convert a `pdaggerq` symbol to the internal format.

    Parameters
    ----------
    symbol : str
        Symbol to convert.
    index_spins : dict, optional
        Dictionary mapping indices to spins. If not provided, the
        indices are not constrained to a single spin. Default value is
        `None`.
    index_spaces : dict, optional
        Dictionary mapping indices to spaces. If not provided, the
        indices are assigned to the spaces based on their names. Default
        value is `None`.

    Returns
    -------
    out : Tensor or Number
        Converted symbol.
    """

    if re.match(r".*_[0-9]+$", symbol):
        # Symbol has spaces
        symbol, spaces = symbol.rsplit("_", 1)

    if _is_number(symbol):
        # factor
        return float(symbol)

    elif re.match(r"f\([a-z],[a-z]\)", symbol):
        # f(i,j)
        indices = tuple(symbol[2:-1].replace("t", "p").split(","))
        tensor_symbol = Fock

    elif re.match(r"<[a-z],[a-z]\|\|[a-z],[a-z]>", symbol):
        # <i,j||k,l>
        indices = tuple(symbol[1:-1].replace("t", "p").replace("||", ",").split(","))
        tensor_symbol = ERI

    elif re.match(r"<[a-z],[a-z]\|[a-z],[a-z]>", symbol):
        # <i,j|k,l>
        indices = tuple(symbol[1:-1].replace("t", "p").replace("|", ",").split(","))
        tensor_symbol = SingleERI

    elif re.match(r"t1\([a-z],[a-z]\)", symbol):
        # t1(i,j)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        indices = (indices[1], indices[0])
        tensor_symbol = T1

    elif re.match(r"t2\([a-z],[a-z],[a-z],[a-z]\)", symbol):
        # t2(i,j,k,l)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        indices = (indices[2], indices[3], indices[0], indices[1])
        tensor_symbol = T2

    elif re.match(r"t3\([a-z],[a-z],[a-z],[a-z],[a-z],[a-z]\)", symbol):
        # t3(i,j,k,l,m,n)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        indices = (indices[3], indices[4], indices[5], indices[0], indices[1], indices[2])
        tensor_symbol = T3

    elif re.match(r"l1\([a-z],[a-z]\)", symbol):
        # l1(i,j)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        indices = (indices[1], indices[0])
        tensor_symbol = L1

    elif re.match(r"l2\([a-z],[a-z],[a-z],[a-z]\)", symbol):
        # l2(i,j,k,l)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        indices = (indices[2], indices[3], indices[0], indices[1])
        tensor_symbol = L2

    elif re.match(r"l3\([a-z],[a-z],[a-z],[a-z],[a-z],[a-z]\)", symbol):
        # l3(i,j,k,l,m,n)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        indices = (indices[3], indices[4], indices[5], indices[0], indices[1], indices[2])
        tensor_symbol = L3

    elif re.match(r"r1\([a-z]\)", symbol):
        # r1(i)
        indices = (symbol[3],)
        #tensor_symbol = R1ip if _to_space(symbol[3]) == "o" else R1ea
        tensor_symbol = R1ip  # FIXME

    elif re.match(r"r2\([a-z],[a-z],[a-z]\)", symbol):
        # r2(i,j,a)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        if _to_space(indices[1]) == "o":
            indices = (indices[1], indices[2], indices[0])
            tensor_symbol = R2ip
        else:
            #indices = (indices[2], indices[0], indices[1])
            #tensor_symbol = R2ea
            tensor_symbol = R2ip  # FIXME

    elif re.match(r"r3\([a-z],[a-z],[a-z],[a-z],[a-z]\)", symbol):
        # r3(i,j,k,a,b)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        if _to_space(indices[2]) == "o":
            indices = (indices[2], indices[3], indices[4], indices[0], indices[1])
            tensor_symbol = R3ip
        else:
            #indices = (indices[3], indices[4], indices[0], indices[1], indices[2])
            #tensor_symbol = R3ea
            tensor_symbol = R3ip  # FIXME

    elif re.match(r"r1\([a-z],[a-z]\)", symbol):
        # r1(a,i)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        indices = (indices[1], indices[0])
        tensor_symbol = R1ee

    elif re.match(r"r2\([a-z],[a-z],[a-z],[a-z]\)", symbol):
        # r2(a,b,i,j)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        indices = (indices[2], indices[3], indices[0], indices[1])
        tensor_symbol = R2ee

    elif re.match(r"r3\([a-z],[a-z],[a-z],[a-z],[a-z],[a-z]\)", symbol):
        # r3(a,b,c,i,j,k)
        indices = tuple(symbol[3:-1].replace("t", "p").split(","))
        indices = (indices[3], indices[4], indices[5], indices[0], indices[1], indices[2])
        tensor_symbol = R3ee

    elif re.match(r"d\([a-z],[a-z]\)", symbol):
        # d(i,j)
        indices = tuple(symbol[2:-1].replace("t", "p").split(","))
        tensor_symbol = Delta

    elif re.match(r"P\([a-z],[a-z]\)", symbol):
        # P(i,j)
        indices = tuple(symbol[2:-1].replace("t", "p").split(","))
        tensor_symbol = P

    else:
        raise ValueError(f"Unknown symbol {symbol}")

    # Convert the indices to Index
    indices = tuple(
        Index(
            index,
            spin=index_spins.get(index),
            space=index_spaces.get(index, _to_space(index)),
        )
        for index in indices
    )

    return tensor_symbol[indices]
