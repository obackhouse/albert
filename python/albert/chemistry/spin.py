"""Spin integration."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

from albert._core import Add, Expression, Mul, Scalar, Tensor, expand, flatten, simplify
from albert.chemistry.options import SUPPORTED_TENSOR_SYMBOLS, OrbitalType, ReferenceType
from albert.chemistry.transform import expand_antisymmetrized_eris

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from albert._core import Node
    from albert.chemistry.metadata import Metadata


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


def _spin_patterns(
    tensor: Tensor,
    *,
    target_rhf: bool = False,
    transform_like: Mapping[str, str] | None = None,
) -> Iterable[tuple[str, ...]]:
    """Enumerate UHF spin blocks allowed for a chemistry tensor."""
    symbol = tensor.symbol if transform_like is None else transform_like.get(tensor.symbol, tensor.symbol)
    rank = tensor.rank

    if rank == 0:
        yield ()
    elif symbol in {"h", "f", "d1"}:
        yield from ((spin, spin) for spin in (OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript))
    elif symbol == "v" or symbol in {"d2", "d2_a"}:
        yield from (
            (first, second, first, second)
            for first in (OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript)
            for second in (OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript)
        )
    elif symbol in {"t1", "t2", "t3", "t2_a", "t3_a", "l1", "l2", "l3", "l2_a", "l3_a"}:
        if rank % 2:
            raise ValueError(f"amplitude tensor {symbol!r} must have even rank")
        half_rank = rank // 2
        spins = (OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript)
        if symbol.endswith("_a") or target_rhf:
            yield from (
                pattern
                for pattern in itertools.product(spins, repeat=rank)
                if sorted(pattern[:half_rank]) == sorted(pattern[half_rank:])
            )
        else:
            yield from (pattern + pattern for pattern in itertools.product(spins, repeat=half_rank))
    else:
        raise NotImplementedError(f"spin integration not implemented for tensor {symbol!r}")


def _allowed_patterns(
    tensor: Tensor,
    *,
    target_rhf: bool = False,
    transform_like: Mapping[str, str] | None = None,
) -> list[tuple[str, ...]]:
    """Filter spin patterns against any spin already carried by the tensor."""
    fixed = [
        spin if spin in (OrbitalType.ALPHA.subscript, OrbitalType.BETA.subscript) else None
        for spin in (_strip_spin(index)[1] for index in tensor.indices)
    ]
    return [
        pattern
        for pattern in _spin_patterns(tensor, target_rhf=target_rhf, transform_like=transform_like)
        if all(old is None or old == new for old, new in zip(fixed, pattern, strict=True))
    ]


def _ghf_tensor_to_uhf(
    tensor: Tensor,
    *,
    target_rhf: bool = False,
    transform_like: Mapping[str, str] | None = None,
) -> list[Tensor]:
    """Return the allowed UHF blocks for one GHF tensor."""
    return [
        _tensor_with_spins(tensor, pattern)
        for pattern in _allowed_patterns(tensor, target_rhf=target_rhf, transform_like=transform_like)
    ]


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


def _ghf_term_to_uhf(
    term: Node,
    *,
    target_rhf: bool = False,
    transform_like: Mapping[str, str] | None = None,
) -> list[Node]:
    """Expand one GHF term into spin-consistent UHF terms."""
    factors = term.children if isinstance(term, Mul) else [term]
    scalar_factors = [factor for factor in factors if isinstance(factor, Scalar)]
    tensor_factors = [factor for factor in factors if isinstance(factor, Tensor)]
    if len(scalar_factors) + len(tensor_factors) != len(factors):
        raise NotImplementedError("spin integration supports only scalar and tensor factors")

    tensor_choices = [
        _ghf_tensor_to_uhf(tensor, target_rhf=target_rhf, transform_like=transform_like) for tensor in tensor_factors
    ]
    if not tensor_choices:
        return [term]

    terms = []
    for tensors in itertools.product(*tensor_choices):
        if not _spin_consistent(tensors):
            continue
        factors = [*scalar_factors, *tensors]
        terms.append(factors[0] if len(factors) == 1 else Mul(factors))
    return terms


def _uhf_tensor_to_rhf(node: Node, transform_like: Mapping[str, str] | None = None) -> Node:
    """Convert a UHF tensor to spatial-orbital form."""
    if not isinstance(node, Tensor):
        return node
    spin_indices = [_strip_spin(index) for index in node.indices]
    spins = tuple(spin for _, spin in spin_indices)
    indices = []
    for (base, _), spin in zip(spin_indices, spins, strict=True):
        if spin == OrbitalType.SPIN.subscript or spin is None:
            raise ValueError(f"expected alpha/beta UHF index, got {base!r}")
        indices.append(base)

    # A spin-orbital doubles amplitude is antisymmetric in each index pair.
    # Its RHF representation is the spin-adapted spatial amplitude. Same-spin
    # blocks therefore contribute the difference of the two virtual orders;
    # opposite-spin blocks contribute one of those orders, with the crossed
    # block carrying the antisymmetry sign.
    symbol = node.symbol if transform_like is None else transform_like.get(node.symbol, node.symbol)
    if symbol in {"t2", "l2"} and node.rank == 4:
        alpha = OrbitalType.ALPHA.subscript
        beta = OrbitalType.BETA.subscript
        swapped = [*indices[:2], indices[3], indices[2]]
        if len(set(spins)) == 1:
            return Add([Tensor(node.symbol, indices), -Tensor(node.symbol, swapped)])
        if spins in {(alpha, beta, alpha, beta), (beta, alpha, beta, alpha)}:
            return Tensor(node.symbol, indices)
        if spins in {(alpha, beta, beta, alpha), (beta, alpha, alpha, beta)}:
            return -Tensor(node.symbol, swapped)
        raise ValueError(f"unsupported UHF doubles spin pattern for {node.symbol!r}: {spins}")

    return Tensor(node.symbol, indices)


def _ghf_to_uhf(
    expression: Expression,
    *,
    target_rhf: bool = False,
    transform_like: Mapping[str, str] | None = None,
) -> list[Node]:
    """Transform and expand an expression's right-hand side to UHF blocks."""
    expanded_antisymmetry = expression.rhs.apply(
        lambda node: expand_antisymmetrized_eris(node, transform_like=transform_like)
    )
    expanded_rhs = simplify(flatten(expand(expanded_antisymmetry)))
    return [
        transformed
        for term in _additive_terms(expanded_rhs)
        for transformed in _ghf_term_to_uhf(term, target_rhf=target_rhf, transform_like=transform_like)
    ]


def _uhf_to_rhf(
    expression: Expression,
    metadata: Metadata | None = None,
    transform_like: Mapping[str, str] | None = None,
) -> Node:
    """Transform and collect the UHF right-hand side in spatial orbitals."""
    expanded_antisymmetry = expression.rhs.apply(
        lambda node: expand_antisymmetrized_eris(node, transform_like=transform_like)
    )
    transformed = simplify(
        flatten(
            expand(expanded_antisymmetry.apply(lambda node: _uhf_tensor_to_rhf(node, transform_like=transform_like)))
        )
    )
    if expression.lhs.rank == 0 and metadata is not None:
        transformed = _collect_scalar_dummy_permutations(transformed, metadata)
    return transformed


def _additive_terms(node: Node) -> list[Node]:
    """Return the additive terms for a node."""
    return node.children if isinstance(node, Add) else [node]


def _normalized_scalar_term(
    term: Node,
    metadata: Metadata,
) -> tuple[tuple[tuple[str, tuple[int, ...]], ...], tuple[str, ...], float, list[str]] | None:
    """Return a tensor pattern, its spaces, coefficient, and source labels."""
    factors = term.children if isinstance(term, Mul) else [term]
    coefficient = math.prod(factor.value for factor in factors if isinstance(factor, Scalar))
    tensors = [factor for factor in factors if isinstance(factor, Tensor)]
    if len(tensors) + sum(isinstance(factor, Scalar) for factor in factors) != len(factors):
        return None
    tensors.sort(key=lambda tensor: tensor.symbol)

    label_ids: dict[str, int] = {}
    label_spaces: list[str] = []
    labels_by_id: list[str] = []
    signature = []
    for tensor in tensors:
        positions = []
        for index in tensor.indices:
            space = next((space for space, indices in metadata.indices.items() if index in indices), None)
            if space is None:
                return None
            if index not in label_ids:
                label_ids[index] = len(label_ids)
                label_spaces.append(space)
                labels_by_id.append(index)
            elif label_spaces[label_ids[index]] != space:
                return None
            positions.append(label_ids[index])
        signature.append((tensor.symbol, tuple(positions)))
    return tuple(signature), tuple(label_spaces), coefficient, labels_by_id


def _collect_scalar_dummy_permutations(node: Node, metadata: Metadata) -> Node:
    """Collect scalar terms that differ only by dummy relabelling within spaces."""
    terms = _additive_terms(flatten(node))
    term_data = [_normalized_scalar_term(term, metadata) for term in terms]
    if any(item is None for item in term_data):
        return node

    # Space tuples are part of the key, so dummy renaming cannot cross spaces.
    groups: dict[
        tuple[tuple[tuple[str, tuple[int, ...]], ...], tuple[str, ...]],
        tuple[float, list[str]],
    ] = {}
    templates: dict[tuple[str, ...], list[str]] = {}
    for item in term_data:
        assert item is not None
        signature, spaces, coefficient, source_labels = item
        templates.setdefault(spaces, source_labels)
        key = (signature, spaces)
        previous, labels_by_id = groups.get(key, (0.0, templates[spaces]))
        groups[key] = (previous + coefficient, labels_by_id)

    collected = []
    for (signature, _), (coefficient, labels_by_id) in groups.items():
        if coefficient == 0.0:
            continue
        factors: list[Node] = [
            Tensor(symbol, [labels_by_id[index] for index in positions]) for symbol, positions in signature
        ]
        if coefficient != 1.0:
            factors.insert(0, Scalar(coefficient))
        collected.append(factors[0] if len(factors) == 1 else Mul(factors))
    if not collected:
        return Scalar(0.0)
    return simplify(collected[0] if len(collected) == 1 else Add(collected))


def _ghf_expression_to_uhf(
    expression: Expression,
    *,
    target_rhf: bool = False,
    transform_like: Mapping[str, str] | None = None,
) -> list[Expression]:
    """Convert a GHF equation to one UHF equation per external spin block."""
    grouped_terms: dict[tuple[str, ...], list[Node]] = {}
    allowed_lhs_patterns = set(_allowed_patterns(expression.lhs, target_rhf=target_rhf, transform_like=transform_like))
    for term in _ghf_to_uhf(expression, target_rhf=target_rhf, transform_like=transform_like):
        external_spins = {_strip_spin(index)[0]: _strip_spin(index)[1] for index in term.external_indices}
        lhs_indices = []
        lhs_pattern = []
        for index in expression.lhs.indices:
            base, _ = _strip_spin(index)
            spin = external_spins.get(base)
            if spin is None:
                break
            lhs_indices.append(base + spin)
            lhs_pattern.append(spin)
        else:
            if tuple(lhs_pattern) in allowed_lhs_patterns:
                grouped_terms.setdefault(tuple(lhs_indices), []).append(term)
    combined = [simplify(flatten(expand(Add(terms)))) for terms in grouped_terms.values()]
    return [
        Expression(Tensor(expression.lhs.symbol, indices), term)
        for indices, term in zip(grouped_terms, combined, strict=True)
    ]


def _uhf_expression_to_rhf(
    expression: Expression,
    metadata: Metadata | None = None,
    transform_like: Mapping[str, str] | None = None,
) -> list[Expression]:
    """Convert a UHF equation to one spatial-orbital equation."""
    transformed_rhs = _uhf_to_rhf(expression, metadata, transform_like)
    terms = _additive_terms(flatten(expand(transformed_rhs)))
    combined = simplify(flatten(expand(Add(terms))))
    return [Expression(Tensor(expression.lhs.symbol, combined.external_indices), combined)]


def _ghf_expression_to_rhf(
    expression: Expression,
    metadata: Metadata | None = None,
    transform_like: Mapping[str, str] | None = None,
) -> list[Expression]:
    """Convert a GHF equation to one spatial-orbital equation."""
    terms = [
        term
        for uhf_expression in _ghf_expression_to_uhf(expression, target_rhf=True, transform_like=transform_like)
        for term in _additive_terms(flatten(expand(_uhf_to_rhf(uhf_expression, metadata, transform_like))))
    ]
    if not terms:
        return []
    combined = simplify(flatten(expand(Add(terms))))
    if expression.lhs.rank == 0 and metadata is not None:
        combined = _collect_scalar_dummy_permutations(combined, metadata)
    return [Expression(Tensor(expression.lhs.symbol, combined.external_indices), combined)]


def _validate_transform_like(
    expression: Expression,
    transform_like: dict[str, str] | None,
) -> dict[str, str]:
    """Validate aliases and ensure aliased tensors match their reference ranks."""
    if transform_like is None:
        return {}
    if not isinstance(transform_like, dict):
        raise TypeError("transform_like must be a dict mapping tensor symbols to chemistry tensor symbols")
    for symbol, reference in transform_like.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("transform_like keys must be non-empty tensor symbols")
        if not isinstance(reference, str) or reference not in SUPPORTED_TENSOR_SYMBOLS:
            raise ValueError(f"unknown transform_like reference tensor {reference!r}")

    stack = [expression.lhs, expression.rhs]
    while stack:
        node = stack.pop()
        if isinstance(node, Tensor):
            reference_symbol = transform_like.get(node.symbol)
            if reference_symbol is not None and node.rank != SUPPORTED_TENSOR_SYMBOLS[reference_symbol].rank:
                raise ValueError(
                    f"tensor {node.symbol!r} has rank {node.rank}, but transform_like reference "
                    f"{reference_symbol!r} has rank {SUPPORTED_TENSOR_SYMBOLS[reference_symbol].rank}"
                )
        else:
            stack.extend(node.children)
    return transform_like


def spin_integrate(
    expression: Expression,
    src: ReferenceType,
    dst: ReferenceType,
    metadata: Metadata | None = None,
    transform_like: dict[str, str] | None = None,
) -> list[Expression]:
    """Perform spin integration to project onto a more restricted reference type.

    Args:
        expression: The expression to transform.
        src: The source reference type.
        dst: The destination reference type.
        metadata: Tensor metadata used to collect scalar contractions by space-preserving dummy relabelling.
        transform_like: Map tensor symbols to known chemistry tensors with matching spin behavior.

    Returns:
        A list of expressions resulting from the transformation.
    """
    src = ReferenceType(src)
    dst = ReferenceType(dst)
    transform_like = _validate_transform_like(expression, transform_like)
    if src == dst:
        return [expression]
    if src is ReferenceType.GHF and dst is ReferenceType.UHF:
        return _ghf_expression_to_uhf(expression, transform_like=transform_like)
    if src is ReferenceType.UHF and dst is ReferenceType.RHF:
        return _uhf_expression_to_rhf(expression, metadata, transform_like)
    if src is ReferenceType.GHF and dst is ReferenceType.RHF:
        return _ghf_expression_to_rhf(expression, metadata, transform_like)
    raise ValueError(f"unsupported spin integration from {src.value} to {dst.value}")
