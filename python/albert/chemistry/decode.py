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


def _pdaggerq_indices(token: str) -> tuple[str, ...]:
    """Return the indices carried by an ordinary pdaggerq tensor token."""
    if token.startswith("<") and token.endswith(">"):
        body = token[1:-1].replace("||", ",").replace("|", ",")
        return _indices(body)
    match = _FUNCTION_TOKEN.fullmatch(token)
    return () if match is None or match.group(1) == "P" else _indices(match.group(2))


def _is_eom_vector_token(token: str, *, include_left: bool) -> bool:
    match = _FUNCTION_TOKEN.fullmatch(token)
    if match is None:
        return False
    symbol = match.group(1)
    return (symbol.startswith("r") or (include_left and symbol.startswith("l"))) and symbol[-1:] in {"1", "2", "3"}


def _is_reference_energy_term(term: list[str], *, include_left: bool) -> bool:
    """Whether one pdaggerq EOM term is a disconnected reference-energy term."""
    body = term[1:] if term and _is_number_token(term[0]) else term
    vectors = [token for token in body if _is_eom_vector_token(token, include_left=include_left)]
    if len(vectors) != 1:
        return False
    other_tensors = [token for token in body if token != vectors[0] and not token.startswith("P(")]
    if set(_pdaggerq_indices(vectors[0])) & {index for token in other_tensors for index in _pdaggerq_indices(token)}:
        return False

    if len(body) == 2 and len(other_tensors) == 1:
        token = other_tensors[0]
        match = _FUNCTION_TOKEN.fullmatch(token)
        fock_energy = (
            match is not None
            and match.group(1) == "f"
            and (indices := _indices(match.group(2), expected=2))[0] == indices[1]
        )
        eri_indices = _pdaggerq_indices(token) if token.startswith("<") and token.endswith(">") else ()
        eri_energy = len(eri_indices) == 4 and eri_indices[0] == eri_indices[2] and eri_indices[1] == eri_indices[3]
        return fock_energy or eri_energy

    tensors = sorted(other_tensors)
    fock_amplitude = len(tensors) >= 2 and tensors[0].startswith("f") and tensors[1].startswith("t")
    eri_amplitudes = (
        len(tensors) >= 2 and tensors[0].startswith("<") and all(token.startswith("t") for token in tensors[1:])
    )
    return fock_amplitude or eri_amplitudes


def _is_number_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _remove_reference_energy_eom(terms: list[list[str]], *, include_left: bool) -> list[list[str]]:
    """Drop disconnected E0 contributions from pdaggerq EOM vector terms."""
    return [term for term in terms if not _is_reference_energy_term(term, include_left=include_left)]


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
    if len(indices) % 2 == 0:
        kind = "ee"
    elif _occupied(indices[rank - 1]):
        kind = "ip"
    elif _virtual(indices[rank - 1]):
        kind = "ea"
    mapped: tuple[str, ...] = indices
    if kind == "ee":
        mapped = (*indices[rank:], *indices[:rank])
    elif kind == "ip":
        if prefix == "r" and rank == 2:
            mapped = (indices[1], indices[2], indices[0])
        elif prefix == "r" and rank == 3:
            mapped = (*indices[2:], *indices[:2])
        else:
            mapped = indices
    elif kind == "ea":
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
    Index spaces and tensor metadata are intentionally left to the caller. Disconnected
    reference-energy terms are removed automatically when the term list contains EOM vectors.
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

        term_lists = [list(term) for term in terms]
        include_left = not self.l_is_lambda
        if any(_is_eom_vector_token(token, include_left=include_left) for term in term_lists for token in term):
            term_lists = _remove_reference_energy_eom(term_lists, include_left=include_left)

        decoded_terms = []
        for term in term_lists:
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
