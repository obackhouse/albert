"""Interface to `pdaggerq`."""

from __future__ import annotations

import re
from numbers import Number
from typing import TYPE_CHECKING

from albert.algebra import _compose_mul
from albert.index import Index
from albert.qc import ghf
from albert.scalar import Scalar
from albert.symmetry import symmetric_group
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Literal, Optional

    from albert.base import Base
    from albert.symmetry import Symmetry


class PermutationOperator(Tensor):
    """Class for a permutation operator.

    Args:
        indices: Indices of the tensor.
        name: Name of the tensor.
        symmetry: Symmetry of the tensor.
    """

    def __init__(
        self,
        *indices: Index,
        name: Optional[str] = None,
        symmetry: Optional[Symmetry] = None,
    ):
        """Initialise the tensor."""
        if len(indices) != 2:
            raise ValueError("Permutation operator must have two indices.")
        if name is None:
            name = "P"
        if symmetry is None:
            symmetry = symmetric_group((0, 1), (1, 0))
        Tensor.__init__(self, *indices, name=name, symmetry=symmetry)


def import_from_pdaggerq(
    terms: list[list[str]],
    index_spins: Optional[dict[str, str]] = None,
    index_spaces: Optional[dict[str, str]] = None,
    l_is_lambda: bool = True,
) -> Base:
    """Import an expression from `pdaggerq`.

    Tensors in the return expression are `GHF` tensors.

    Args:
        terms: The terms of the expression. Should be the output of the `fully_contracted_strings`
            method in `pdaggerq`.
        index_spins: The index spins.
        index_spaces: The index spaces.
        l_is_lambda: Whether `l` corresponds to the Lambda operator, rather than the left-hand EOM
            operator.

    Returns:
        The imported expression.
    """
    if index_spins is None:
        index_spins = {}
    if index_spaces is None:
        index_spaces = {}

    # Build the expression
    expr: Base = Scalar(0.0)
    for term in terms:
        # Convert the symbols
        symbols = [
            _convert_symbol(
                symbol,
                index_spins=index_spins,
                index_spaces=index_spaces,
                l_is_lambda=l_is_lambda,
            )
            for symbol in term
        ]

        # Remove the permutation operators
        perm_ops = filter(lambda symbol: isinstance(symbol, PermutationOperator), symbols)
        symbols = filter(lambda symbol: not isinstance(symbol, PermutationOperator), symbols)
        part = _compose_mul(*symbols)
        for perm_op in perm_ops:
            index_map = {
                perm_op.external_indices[0]: perm_op.external_indices[1],
                perm_op.external_indices[1]: perm_op.external_indices[0],
            }
            part = part - part.map_indices(index_map)

        # Add the term to the expression
        expr += part

    return expr


def _guess_space(index: str, which: Literal["full", "active", "inactive"] = "full") -> str:
    """Guess the space of an index.

    Args:
        index: The index.
        which: The type of the space.
    """
    if index in ("i", "j", "k", "l", "m", "n", "o", "t") or index.startswith("o"):
        if which == "full":
            return "o"
        elif which == "active":
            return "O"
        elif which == "inactive":
            return "i"
    elif index in ("a", "b", "c", "d", "e", "f", "g", "h") or index.startswith("v"):
        if which == "full":
            return "v"
        elif which == "active":
            return "V"
        elif which == "inactive":
            return "a"

    raise ValueError(f"Could not guess space of index {index}.")


def _is_number(obj: Any) -> bool:
    """Check if something is a number.

    Args:
        obj: The object to check.

    Returns:
        Whether the object is a number.
    """
    if isinstance(obj, Number):
        return True
    else:
        try:
            float(obj)
            return True
        except ValueError:
            return False


def _convert_symbol(
    symbol: str,
    index_spins: Optional[dict[str, str]] = None,
    index_spaces: Optional[dict[str, str]] = None,
    l_is_lambda: bool = True,
) -> Base:
    """Convert a symbol to a subclass of `Base`.

    Args:
        symbol: The symbol.
        index_spins: The index spins.
        index_spaces: The index spaces.
        l_is_lambda: Whether `l` corresponds to the Lambda operator, rather than the left-hand EOM
            operator.

    Returns:
        The converted symbol.
    """
    if index_spins is None:
        index_spins = {}
    if index_spaces is None:
        index_spaces = {}

    if re.match(r".*_[0-9]+$", symbol):
        # Symbol has spaces attached, separate them
        symbol, spaces = symbol.rsplit("_", 1)

    if _is_number(symbol):
        # It's the factor
        return Scalar(float(symbol))

    tensor_symbol: type[Tensor]
    index_strs: tuple[str, ...]
    if symbol in ("r0", "l0"):
        # r0 or l0
        index_strs = ()
        tensor_symbol = ghf.R0
    elif re.match(r"f\((?i:[a-z]),(?i:[a-z])\)", symbol):
        # f(i,j)
        index_strs = tuple(symbol[2:-1].split(","))
        tensor_symbol = ghf.Fock
    elif re.match(r"<(?i:[a-z]),(?i:[a-z])\|\|(?i:[a-z]),(?i:[a-z])>", symbol):
        # <i,j||k,l>
        index_strs = tuple(symbol[1:-1].replace("||", ",").split(","))
        tensor_symbol = ghf.ERI
    elif re.match(r"t1\((?i:[a-z]),(?i:[a-z])\)", symbol):
        # t1(i,j)
        index_strs = tuple(symbol[3:-1].split(","))
        index_strs = (index_strs[1], index_strs[0])
        tensor_symbol = ghf.T1
    elif re.match(r"t2\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol):
        # t2(i,j,k,l)
        index_strs = tuple(symbol[3:-1].split(","))
        index_strs = (index_strs[2], index_strs[3], index_strs[0], index_strs[1])
        tensor_symbol = ghf.T2
    elif re.match(
        r"t3\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol
    ):
        # t3(i,j,k,l,m,n)
        index_strs = tuple(symbol[3:-1].split(","))
        index_strs = (
            index_strs[3],
            index_strs[4],
            index_strs[5],
            index_strs[0],
            index_strs[1],
            index_strs[2],
        )
        tensor_symbol = ghf.T3
    elif re.match(r"l1\((?i:[a-z]),(?i:[a-z])\)", symbol) and l_is_lambda:
        # l1(i,j)
        index_strs = tuple(symbol[3:-1].split(","))
        index_strs = (index_strs[1], index_strs[0])
        tensor_symbol = ghf.L1
    elif re.match(r"l2\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol) and l_is_lambda:
        # l2(i,j,k,l)
        index_strs = tuple(symbol[3:-1].split(","))
        index_strs = (index_strs[2], index_strs[3], index_strs[0], index_strs[1])
        tensor_symbol = ghf.L2
    elif (
        re.match(r"l3\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol)
        and l_is_lambda
    ):
        # l3(i,j,k,l,m,n)
        index_strs = tuple(symbol[3:-1].split(","))
        index_strs = (
            index_strs[3],
            index_strs[4],
            index_strs[5],
            index_strs[0],
            index_strs[1],
            index_strs[2],
        )
        tensor_symbol = ghf.L3
    elif re.match(r"r1\((?i:[a-z])\)", symbol):
        # r1(i)
        index_strs = (symbol[3],)
        tensor_symbol = ghf.R1ip  # FIXME: Use EA
    elif re.match(r"r2\((?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol):
        # r2(i,j,a)
        index_strs = tuple(symbol[3:-1].split(","))
        if _guess_space(index_strs[1]) == "o":
            index_strs = (index_strs[1], index_strs[2], index_strs[0])
            tensor_symbol = ghf.R2ip
        else:
            tensor_symbol = ghf.R2ip  # FIXME: Use EA
    elif re.match(r"r3\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol):
        # r3(i,j,k,a,b)
        index_strs = tuple(symbol[3:-1].split(","))
        if _guess_space(index_strs[2]) == "o":
            index_strs = (index_strs[2], index_strs[3], index_strs[4], index_strs[0], index_strs[1])
            tensor_symbol = ghf.R3ip
        else:
            tensor_symbol = ghf.R3ip  # FIXME: Use EA
    elif re.match(r"r1\((?i:[a-z]),(?i:[a-z])\)", symbol):
        # r1(a,i)
        index_strs = tuple(symbol[3:-1].split(","))
        index_strs = (index_strs[1], index_strs[0])
        tensor_symbol = ghf.R1ee
    elif re.match(r"r2\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol):
        # r2(a,b,i,j)
        index_strs = tuple(symbol[3:-1].split(","))
        index_strs = (index_strs[2], index_strs[3], index_strs[0], index_strs[1])
        tensor_symbol = ghf.R2ee
    elif re.match(
        r"r3\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol
    ):
        # r3(a,b,c,i,j,k)
        index_strs = tuple(symbol[3:-1].split(","))
        index_strs = (
            index_strs[3],
            index_strs[4],
            index_strs[5],
            index_strs[0],
            index_strs[1],
            index_strs[2],
        )
        tensor_symbol = ghf.R3ee
    elif re.match(r"l1\((?i:[a-z])\)", symbol) and not l_is_lambda:
        # l1(i)
        index_strs = (symbol[3],)
        tensor_symbol = ghf.R1ip
    elif re.match(r"l2\((?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol):
        # l2(i,j,a)
        index_strs = tuple(symbol[3:-1].split(","))
        if _guess_space(index_strs[1]) == "o":
            tensor_symbol = ghf.R2ip
        else:
            index_strs = (index_strs[1], index_strs[2], index_strs[0])
            tensor_symbol = ghf.R2ip
    elif (
        re.match(r"l3\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol)
        and not l_is_lambda
    ):
        # l3(i,j,k,a,b)
        index_strs = tuple(symbol[3:-1].split(","))
        if _guess_space(index_strs[2]) == "o":
            tensor_symbol = ghf.R3ip
        else:
            index_strs = (index_strs[2], index_strs[3], index_strs[4], index_strs[0], index_strs[1])
            tensor_symbol = ghf.R3ip
    elif re.match(r"l1\((?i:[a-z]),(?i:[a-z])\)", symbol) and not l_is_lambda:
        # l1(i,a)
        index_strs = tuple(symbol[3:-1].split(","))
        tensor_symbol = ghf.R1ee
    elif re.match(r"l2\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol) and not l_is_lambda:
        # l2(i,j,a,b)
        index_strs = tuple(symbol[3:-1].split(","))
        tensor_symbol = ghf.R2ee
    elif (
        re.match(r"l3\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol)
        and not l_is_lambda
    ):
        # l3(i,j,k,a,b,c)
        index_strs = tuple(symbol[3:-1].split(","))
        tensor_symbol = ghf.R3ee
    elif re.match(r"d\((?i:[a-z]),(?i:[a-z])\)", symbol):
        # d(i,j)
        index_strs = tuple(symbol[2:-1].split(","))
        tensor_symbol = ghf.Delta
    elif re.match(r"P\((?i:[a-z]),(?i:[a-z])\)", symbol):
        # P(i,j)
        index_strs = tuple(symbol[2:-1].split(","))
        tensor_symbol = PermutationOperator
    else:
        raise ValueError(f"Unknown symbol {symbol}")

    # Convert the indices
    indices = tuple(
        Index(
            index,
            spin=index_spins.get(index, None),
            space=index_spaces.get(index, _guess_space(index)),
        )
        for index in index_strs
    )

    return tensor_symbol(*indices)


def remove_reference_energy(terms: list[list[str]]) -> list[list[str]]:
    """Remove the reference energy from the terms.

    Args:
        terms: The terms.

    Returns:
        The terms with the reference energy removed.
    """
    terms_new: list[list[str]] = []
    for term in terms:
        if term[0].startswith("+1.0") and term[1] == "f(i,i)":
            continue
        if term[0].startswith("-0.5") and term[1] == "<j,i||j,i>":
            continue
        terms_new.append(term)
    return terms_new
