"""Spin integration."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from albert._core import Add, Expression, Mul, Scalar, Tensor, expand, flatten, simplify
from albert.chemistry.options import OrbitalType, ReferenceType
from albert.chemistry.transform import expand_antisymmetrized_eris

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from albert._core import Node


def _strip_spin(index: str) -> tuple[str, str | None]:
    """Return an index's base name and spin suffix, if it has one."""
    for suffix in (*(OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript), OrbitalType.SPIN.subscript):
        if index.endswith(suffix):
            return index[: -len(suffix)], suffix
    return index, None


def _tensor_with_spins(tensor: Tensor, spins: Sequence[str]) -> Tensor:
    """Return a tensor whose indices have the requested spin labels."""
    indices = []
    for index, spin in zip(tensor.indices, spins, strict=True):
        base, _ = _strip_spin(index)
        indices.append(base + spin)
    return Tensor(tensor.symbol, indices)


def _spin_patterns(tensor: Tensor) -> Iterable[tuple[str, ...]]:
    """Enumerate UHF spin blocks allowed for a chemistry tensor."""
    symbol = tensor.symbol
    rank = tensor.rank

    if symbol in {"h", "f", "d1"}:
        yield from ((spin, spin) for spin in (OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript))
    elif symbol == "v" or symbol in {"d2", "d2_a"}:
        yield from (
            (first, second, first, second)
            for first in (OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript)
            for second in (OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript)
        )
    elif symbol.startswith(("t", "l")) and symbol[1:2].isdigit():
        half_rank = rank // 2
        yield from (
            spins + spins
            for spins in itertools.product((OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript), repeat=half_rank)
        )
    else:
        raise NotImplementedError(f"spin integration not implemented for tensor {symbol!r}")


def _allowed_patterns(tensor: Tensor) -> list[tuple[str, ...]]:
    """Filter spin patterns against any spin already carried by the tensor."""
    fixed = [
        spin if spin in (OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript) else None
        for spin in (_strip_spin(index)[1] for index in tensor.indices)
    ]
    return [
        pattern
        for pattern in _spin_patterns(tensor)
        if all(old is None or old == new for old, new in zip(fixed, pattern, strict=True))
    ]


def _ghf_tensor_to_uhf(tensor: Tensor) -> list[Tensor]:
    """Return the allowed UHF blocks for one GHF tensor."""
    return [_tensor_with_spins(tensor, pattern) for pattern in _allowed_patterns(tensor)]


def _spin_consistent(nodes: Sequence[Node]) -> bool:
    """Return whether repeated index names have one consistent spin."""
    index_spins: dict[str, str] = {}
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, Tensor):
            for index in node.indices:
                base, spin = _strip_spin(index)
                if spin in (OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript):
                    previous = index_spins.setdefault(base, spin)
                    if previous != spin:
                        return False
        else:
            stack.extend(node.children)
    return True


def _ghf_term_to_uhf(term: Node) -> list[Node]:
    """Expand one GHF term into spin-consistent UHF terms."""
    factors = term.children if isinstance(term, Mul) else [term]
    scalar_factors = [factor for factor in factors if isinstance(factor, Scalar)]
    tensor_factors = [factor for factor in factors if isinstance(factor, Tensor)]
    if len(scalar_factors) + len(tensor_factors) != len(factors):
        raise NotImplementedError("spin integration supports only scalar and tensor factors")

    tensor_choices = [_ghf_tensor_to_uhf(tensor) for tensor in tensor_factors]
    if not tensor_choices:
        return [term]

    terms = []
    for tensors in itertools.product(*tensor_choices):
        if not _spin_consistent(tensors):
            continue
        factors = [*scalar_factors, *tensors]
        terms.append(factors[0] if len(factors) == 1 else Mul(factors))
    return terms


def _uhf_tensor_to_rhf(node: Node) -> Node:
    """Remove alpha/beta suffixes from a UHF tensor."""
    if not isinstance(node, Tensor):
        return node
    indices = []
    for index in node.indices:
        base, spin = _strip_spin(index)
        if spin == OrbitalType.SPIN.subscript or spin is None:
            raise ValueError(f"expected alpha/beta UHF index, got {index!r}")
        indices.append(base)
    return Tensor(node.symbol, indices)


def _ghf_to_uhf(expression: Expression) -> list[Node]:
    """Transform and expand an expression's right-hand side to UHF blocks."""
    expanded_antisymmetry = expression.rhs.apply(expand_antisymmetrized_eris)
    expanded_rhs = simplify(flatten(expand(expanded_antisymmetry)))
    return [transformed for term in _additive_terms(expanded_rhs) for transformed in _ghf_term_to_uhf(term)]


def _uhf_to_rhf(expression: Expression) -> Node:
    """Transform and collect the UHF right-hand side in spatial orbitals."""
    expanded_antisymmetry = expression.rhs.apply(expand_antisymmetrized_eris)
    return simplify(flatten(expand(expanded_antisymmetry.apply(_uhf_tensor_to_rhf))))


def _additive_terms(node: Node) -> list[Node]:
    """Return the additive terms for a node."""
    return node.children if isinstance(node, Add) else [node]


def _ghf_expression_to_uhf(expression: Expression) -> list[Expression]:
    """Convert a GHF equation to one UHF equation per external spin block."""
    grouped_terms: dict[tuple[str, ...], list[Node]] = {}
    for term in _ghf_to_uhf(expression):
        grouped_terms.setdefault(tuple(term.external_indices), []).append(term)
    combined = [simplify(flatten(expand(Add(terms)))) for terms in grouped_terms.values()]
    return [Expression(Tensor(expression.lhs.symbol, term.external_indices), term) for term in combined]


def _uhf_expression_to_rhf(expression: Expression) -> list[Expression]:
    """Convert a UHF equation to one spatial-orbital equation."""
    transformed_rhs = _uhf_to_rhf(expression)
    terms = _additive_terms(flatten(expand(transformed_rhs)))
    combined = simplify(flatten(expand(Add(terms))))
    return [Expression(Tensor(expression.lhs.symbol, combined.external_indices), combined)]


def _ghf_expression_to_rhf(expression: Expression) -> list[Expression]:
    """Convert a GHF equation to one spatial-orbital equation."""
    terms = [
        term
        for uhf_expression in _ghf_expression_to_uhf(expression)
        for term in _additive_terms(flatten(expand(_uhf_to_rhf(uhf_expression))))
    ]
    if not terms:
        return []
    combined = simplify(flatten(expand(Add(terms))))
    return [Expression(Tensor(expression.lhs.symbol, combined.external_indices), combined)]


def spin_integrate(
    expression: Expression,
    src: ReferenceType,
    dst: ReferenceType,
) -> list[Expression]:
    """Perform spin integration to project onto a more restricted reference type.

    Args:
        expression: The expression to transform.
        src: The source reference type.
        dst: The destination reference type.

    Returns:
        A list of expressions resulting from the transformation.
    """
    src = ReferenceType(src)
    dst = ReferenceType(dst)
    if src == dst:
        return [expression]
    if src is ReferenceType.GHF and dst is ReferenceType.UHF:
        return _ghf_expression_to_uhf(expression)
    if src is ReferenceType.UHF and dst is ReferenceType.RHF:
        return _uhf_expression_to_rhf(expression)
    if src is ReferenceType.GHF and dst is ReferenceType.RHF:
        return _ghf_expression_to_rhf(expression)
    raise ValueError(f"unsupported spin integration from {src.value} to {dst.value}")
