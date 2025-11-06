"""Interface to `wick`."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from albert.algebra import Mul
from albert.index import Index
from albert.qc import ghf
from albert.qc._pdaggerq import _guess_space, _is_number
from albert.qc.tensor import QTensor
from albert.scalar import Scalar

if TYPE_CHECKING:
    from typing import Optional

    from albert.base import Base


def import_from_wick(
    terms: list[str],
    index_spins: Optional[dict[str, str]] = None,
    index_spaces: Optional[dict[str, str]] = None,
    l_is_lambda: bool = True,
    symbol_aliases: Optional[dict[str, str]] = None,
) -> Base:
    r"""Import an expression from `wick`.

    Tensors in the return expression are `GHF` tensors.

    Args:
        terms: The terms of the expression. Should be the lines of the `repr` of the output
            `AExpression` in `wick`, i.e. `str(AExpression(Ex=...)).split("\n")`.
        index_spins: The index spins.
        index_spaces: The index spaces.
        l_is_lambda: Whether `l` corresponds to the Lambda operator, rather than the left-hand EOM
            operator.
        symbol_aliases: Aliases for symbols.

    Returns:
        The imported expression.
    """
    if index_spins is None:
        index_spins = {}
    if index_spaces is None:
        index_spaces = {}

    # Build the expression
    expr: Base = Scalar.factory(0.0)
    for term_str in terms:
        # Convert the symbols
        term = _split_term(term_str)
        term, names = zip(*[_format_symbol(symbol, aliases=symbol_aliases) for symbol in term])
        symbols = [
            _convert_symbol(
                symbol,
                index_spins=index_spins,
                index_spaces=index_spaces,
                l_is_lambda=l_is_lambda,
                name=name,
            )
            for symbol, name in zip(term, names)
        ]
        part = Mul.factory(*symbols)

        # Add the term to the expression
        expr += part.canonicalise(indices=True)  # wick doesn't guarantee same external indices

    return expr


def _split_term(term: str) -> list[str]:
    """Split a term into its symbols."""
    term = term.lstrip(" ")
    term = term.replace(" ", "")
    term = term.replace("}", "} ").rstrip(" ")
    if r"\sum_{" in term:
        term = re.sub(r"\\sum_\{[^\}]*\}", "", term)
    else:
        i = 0
        while term[i] in "-+0123456789.":
            i += 1
        if i > 0:
            term = term[:i] + " " + term[i:]
    return term.split(" ")


def _format_symbol(symbol: str, aliases: dict[str, str] | None = None) -> tuple[str, str]:
    """Rewrite a `wick` symbol to look like a `pdaggerq` symbol."""
    symbol = re.sub(
        r"([a-zA-Z0-9]+)_\{([^\}]*)\}", lambda m: f"{m.group(1)}({','.join(m.group(2))})", symbol
    )
    symbol_name, indices = symbol.split("(", 1) if "(" in symbol else (symbol, None)
    if aliases is not None:
        symbol_alias = aliases.get(symbol_name, symbol_name)
        symbol = f"{symbol_alias}({indices}" if indices is not None else symbol_alias
    return symbol, symbol_name


def _convert_symbol(
    symbol: str,
    index_spins: Optional[dict[str, str]] = None,
    index_spaces: Optional[dict[str, str]] = None,
    l_is_lambda: bool = True,
    name: str | None = None,
) -> Base:
    """Convert a symbol to a subclass of `Base`.

    Args:
        symbol: The symbol.
        index_spins: The index spins.
        index_spaces: The index spaces.
        l_is_lambda: Whether `l` corresponds to the Lambda operator, rather than the left-hand EOM
            operator.
        name: The name of the tensor.

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
        return Scalar.factory(float(symbol))

    tensor_symbol: type[QTensor]
    index_strs: tuple[str, ...]
    if symbol in ("r0", "l0"):
        # r0 or l0
        index_strs = ()
        tensor_symbol = ghf.R0
    elif re.match(r"f\((?i:[a-z]),(?i:[a-z])\)", symbol):
        # f(i,j)
        index_strs = tuple(symbol[2:-1].split(","))
        tensor_symbol = ghf.Fock
    elif re.match(r"v\((?i:[a-z]),(?i:[a-z]),(?i:[a-z]),(?i:[a-z])\)", symbol):
        # v(i,j,k,l)
        index_strs = tuple(symbol[2:-1].split(","))
        index_strs = (index_strs[2], index_strs[3], index_strs[0], index_strs[1])
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
        index_strs = tuple(symbol[3:-1].split(","))[::-1]
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
        index_strs = tuple(symbol[3:-1].split(","))[::-1]
        index_strs = (
            index_strs[3],
            index_strs[4],
            index_strs[5],
            index_strs[0],
            index_strs[1],
            index_strs[2],
        )
        tensor_symbol = ghf.L3
    elif re.match(r"delta\((?i:[a-z]),(?i:[a-z])\)", symbol):
        # delta(i,j)
        index_strs = tuple(symbol[6:-1].split(","))
        tensor_symbol = ghf.Delta
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

    return tensor_symbol.factory(*indices, name=name)
