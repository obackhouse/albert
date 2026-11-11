"""Decoder for serialized pdaggerq term lists."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from albert._core import Add, Expression, Mul, Scalar, Tensor
from albert.chemistry.factory import TensorFactory
from albert.decode import Decoder

if TYPE_CHECKING:
    from typing import Any

    from albert._core import Node


_INDEX_LIST = r"[A-Za-z](?:,[A-Za-z])*"
_FUNCTION_TOKEN = re.compile(r"([A-Za-z][A-Za-z0-9]*)(?:_\d+)?\(([^()]*)\)")


@dataclass(frozen=True)
class _PermutationOperator:
    """A pdaggerq index permutation operator, expanded over its term."""

    first: str
    second: str


def _literal(value: str) -> Any:
    """Parse JSON or Python-literal serialized data without evaluating code."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError("invalid pdaggerq serialization") from error


def _indices(value: str, *, expected: int | None = None) -> tuple[str, ...]:
    """Parse pdaggerq's comma-separated, single-letter indices."""
    if not re.fullmatch(_INDEX_LIST, value):
        raise ValueError(f"invalid pdaggerq tensor indices: {value!r}")
    indices = tuple(value.split(","))
    if expected is not None and len(indices) != expected:
        raise ValueError(f"expected {expected} indices, got {len(indices)}")
    return indices


def _occupied(index: str) -> bool:
    """Guess whether a conventional pdaggerq index is occupied."""
    return index.lower() in set("ijklmnost")


def _virtual(index: str) -> bool:
    """Guess whether a conventional pdaggerq index is virtual."""
    return index.lower() in set("abcdefgh")


def _parse_amplitude(symbol: str, indices: tuple[str, ...]) -> Tensor:
    """Reorder a pdaggerq excitation amplitude into chemistry slot order."""
    permutations = {
        "1": (1, 0),
        "2": (2, 3, 0, 1),
        "3": (3, 4, 5, 0, 1, 2),
    }
    rank_key = symbol[-1:]
    permutation = permutations.get(rank_key)
    if permutation is None or len(indices) != len(permutation):
        raise ValueError(f"unsupported pdaggerq tensor: {symbol}({','.join(indices)})")
    mapped = tuple(indices[position] for position in permutation)
    return TensorFactory(symbol)[mapped]


def _parse_open_shell_eom(symbol: str, indices: tuple[str, ...]) -> Tensor:
    """Map ionization/attachment EOM tokens to the corresponding chemistry tensor."""
    prefix = symbol[0]
    rank = int(symbol[1:])
    occupied = all(_occupied(index) for index in indices[:rank])
    virtual = all(_virtual(index) for index in indices[:rank])
    mapped: tuple[str, ...] = indices
    if occupied:
        kind = "ip"
        if prefix == "r" and rank == 2:
            mapped = (indices[1], indices[2], indices[0])
        elif prefix == "r" and rank == 3:
            mapped = (*indices[2:], *indices[:2])
        else:
            mapped = indices
    elif virtual:
        kind = "ea"
        mapped = indices
    else:
        raise ValueError(f"cannot classify pdaggerq tensor indices in {symbol}({','.join(indices)})")
    return TensorFactory(f"{prefix}{rank}{kind}")[mapped]


def _parse_excitation_eom(symbol: str, indices: tuple[str, ...]) -> Tensor:
    """Map particle-hole EOM tokens to the chemistry tensor convention."""
    prefix = symbol[0]
    rank = int(symbol[1:])
    mapped: tuple[str, ...] = indices
    if prefix == "r":
        mapped = (indices[1], indices[0]) if rank == 1 else (*indices[rank:], *indices[:rank])
    return TensorFactory(f"{prefix}{rank}ee")[mapped]


def _parse_eom_tensor(symbol: str, indices: tuple[str, ...]) -> Tensor:
    """Map pdaggerq's EOM tokens to ionization, attachment, or excitation tensors."""
    rank = int(symbol[1:])
    if len(indices) == 2 * rank:
        return _parse_excitation_eom(symbol, indices)
    if len(indices) == 2 * rank - 1:
        return _parse_open_shell_eom(symbol, indices)
    raise ValueError(f"unsupported pdaggerq tensor: {symbol}({','.join(indices)})")


def _parse_eri_token(token: str) -> Tensor:
    """Decode a pdaggerq integral token using physicists' index order."""
    body = token[1:-1]
    if "||" in body:
        left, right = body.split("||", maxsplit=1)
        return TensorFactory("v_a")[_indices(left, expected=2) + _indices(right, expected=2)]
    if "|" in body:
        left, right = body.split("|", maxsplit=1)
        return TensorFactory("v")[_indices(left, expected=2) + _indices(right, expected=2)]
    raise ValueError(f"invalid pdaggerq ERI token: {token!r}")


@dataclass(frozen=True)
class PdaggerqDecoder(Decoder):
    """Decode pdaggerq's ``list[list[str]]`` term representation.

    ``decode_node`` accepts the string representation of the terms returned by
    pdaggerq, for example ``[["+1.0", "f(i,j)"]]``. It uses Python's safe
    literal parser and does not import or require pdaggerq.

    ``decode_expression`` accepts a serialized mapping with an LHS tensor token
    and a pdaggerq term list, such as
    ``{"lhs": "t1(a,i)", "terms": [["+1.0", "f(i,a)"]]}``.
    Index spaces and tensor metadata are intentionally left to the caller.
    """

    l_is_lambda: bool = True
    """Interpret pdaggerq ``lN`` tensors as Lambda amplitudes when true."""

    def __post_init__(self) -> None:
        if not isinstance(self.l_is_lambda, bool):
            raise ValueError("l_is_lambda must be a bool")

    def _parse_tensor(self, symbol: str, indices: tuple[str, ...]) -> Tensor:
        """Map a pdaggerq tensor token to the chemistry tensor convention."""
        if symbol == "f":
            return TensorFactory("f")[indices]
        if symbol in {"t1", "t2", "t3"} or (self.l_is_lambda and symbol in {"l1", "l2", "l3"}):
            return _parse_amplitude(symbol, indices)
        if symbol in {"r1", "r2", "r3"} or symbol in {"l1", "l2", "l3"}:
            return _parse_eom_tensor(symbol, indices)
        else:
            raise ValueError(f"unsupported pdaggerq tensor: {symbol}({','.join(indices)})")

    def _parse_token(self, token: str) -> Node | _PermutationOperator:
        """Convert one pdaggerq token to a node or permutation operator."""
        token = token.strip()
        try:
            value = float(token)
        except ValueError:
            value = None
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("pdaggerq scalar factors must be finite")
            return Scalar(value)

        if token in {"r0", "l0"}:
            node: Node | _PermutationOperator = TensorFactory(token)[()]
        elif token.startswith("<") and token.endswith(">"):
            node = _parse_eri_token(token)
        else:
            node = self._parse_function_token(token)
        return node

    def _parse_function_token(self, token: str) -> Node | _PermutationOperator:
        """Decode an indexed pdaggerq token."""
        match = _FUNCTION_TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(f"unsupported pdaggerq token: {token!r}")
        symbol, raw_indices = match.groups()
        indices = _indices(raw_indices)
        if symbol == "P":
            if len(indices) != 2:
                raise ValueError("pdaggerq permutation operators require two indices")
            return _PermutationOperator(*indices)
        if symbol == "d":
            if len(indices) != 2:
                raise ValueError("pdaggerq delta tensors require two indices")
            return Tensor("δ", indices)
        return self._parse_tensor(symbol, indices)

    @staticmethod
    def _product(factors: list[Node]) -> Node:
        factors = [factor for factor in factors if not isinstance(factor, Scalar) or factor.value != 1.0]
        if not factors:
            return Scalar(1.0)
        return factors[0] if len(factors) == 1 else Mul(factors)

    def _decode_terms(self, terms: Any) -> Node:
        if not isinstance(terms, (list, tuple)) or any(not isinstance(term, (list, tuple)) for term in terms):
            raise ValueError("pdaggerq terms must be a list of token lists")

        decoded_terms = []
        for term in terms:
            if any(not isinstance(token, str) for token in term):
                raise ValueError("pdaggerq term tokens must be strings")
            factors: list[Node] = []
            permutations: list[_PermutationOperator] = []
            for token in term:
                parsed = self._parse_token(token)
                if isinstance(parsed, _PermutationOperator):
                    permutations.append(parsed)
                else:
                    factors.append(parsed)
            node = self._product(factors)
            for permutation in permutations:
                index_map = {permutation.first: permutation.second, permutation.second: permutation.first}
                node = Add([node, -node.map_indices(index_map)])
            decoded_terms.append(node)

        if not decoded_terms:
            return Scalar(0.0)
        return decoded_terms[0] if len(decoded_terms) == 1 else Add(decoded_terms)

    def decode_node(self, value: str) -> Node:
        """Decode pdaggerq terms as an algebra node."""
        return self._decode_terms(_literal(value))

    def decode_expression(self, value: str) -> Expression:
        """Decode an object containing ``lhs`` and pdaggerq ``terms`` fields."""
        payload = _literal(value)
        if not isinstance(payload, dict) or set(payload) != {"lhs", "terms"}:
            raise ValueError("pdaggerq expression must contain exactly 'lhs' and 'terms'")
        if not isinstance(payload["lhs"], str):
            raise ValueError("pdaggerq expression lhs must be a tensor token")
        lhs_token = payload["lhs"]
        try:
            lhs = self._parse_token(lhs_token)
        except ValueError:
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", lhs_token):
                lhs = Tensor(lhs_token, [])
            else:
                raise
        if not isinstance(lhs, Tensor):
            raise ValueError("pdaggerq expression lhs must decode to a tensor")
        return Expression(lhs, self._decode_terms(payload["terms"]))

    def __call__(self, value: str) -> Node | Expression:
        """Decode pdaggerq terms or an expression envelope."""
        payload = _literal(value)
        if isinstance(payload, dict):
            return self.decode_expression(value)
        return self._decode_terms(payload)
