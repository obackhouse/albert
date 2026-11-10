import pytest

from albert import Add, Expression, Metadata, Mul, Permutation, PhaseFactor, Scalar, Tensor, canonical_key, canonicalize


def test_canonicalize_sorts_and_flattens_additions_and_products() -> None:
    a = Tensor("A", ["i"])
    b = Tensor("B", ["i"])
    c = Tensor("C", ["i"])

    assert canonicalize(Add([c, Add([b, a])])) == canonicalize(Add([a, b, c]))
    product = Mul([Tensor("C", ["k"]), Mul([Tensor("B", ["j"]), Tensor("A", ["i"])])])
    reordered_product = Mul([Tensor("A", ["i"]), Tensor("B", ["j"]), Tensor("C", ["k"])])
    assert canonicalize(product) == canonicalize(reordered_product)


def test_canonicalize_relabels_external_indices_within_a_space() -> None:
    metadata = Metadata(indices={"occupied": ["i", "j", "k", "x", "y", "z"]}, sizes={"occupied": 8})
    left = Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j", "k"])])
    renamed = Mul([Tensor("A", ["x", "z"]), Tensor("B", ["z", "y"])])
    assert canonicalize(left, metadata) == canonicalize(renamed, metadata)


def test_canonicalize_expression_remaps_lhs_external_indices() -> None:
    metadata = Metadata(indices={"occupied": ["i", "j", "k"]}, sizes={"occupied": 8})
    expression = Expression(Tensor("Y", ["k"]), Tensor("A", ["k"]))

    canonical = canonicalize(expression, metadata)

    assert isinstance(canonical, Expression)
    assert canonical.lhs.symbol == "Y"
    assert canonical.lhs.indices == ["i"]
    assert canonical.rhs.external_indices == canonical.lhs.external_indices


def test_canonicalize_expression_preserves_lhs_axis_order_while_remapping() -> None:
    metadata = Metadata(indices={"occupied": ["i", "j", "k", "l"]}, sizes={"occupied": 8})
    expression = Expression(Tensor("Y", ["l", "k"]), Tensor("A", ["l", "k"]))

    canonical = canonicalize(expression, metadata)

    assert isinstance(canonical, Expression)
    assert canonical.lhs.indices == canonical.rhs.indices
    assert canonical.lhs.indices == ["i", "j"]


def test_canonicalize_reserves_canonical_external_labels_before_dummy_labels() -> None:
    metadata = Metadata(indices={"occupied": ["i", "j", "k", "l", "x", "y", "z"]}, sizes={"occupied": 8})
    expression = Mul([Tensor("A", ["j", "k"]), Tensor("B", ["k", "i"])])
    renamed = Mul([Tensor("A", ["x", "z"]), Tensor("B", ["z", "y"])])

    canonical = canonicalize(expression, metadata)

    assert set(canonical.external_indices) == {"i", "j"}
    assert set(canonical.internal_indices).isdisjoint(canonical.external_indices)
    assert canonical == canonicalize(renamed, metadata)


def test_canonicalize_is_independent_of_internal_label_and_factor_order() -> None:
    metadata = Metadata(indices={"all": ["i", "j", "k", "l", "x", "y", "z"]}, sizes={"all": 8})
    left = Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k", "j"]), Tensor("C", ["j", "i"])])
    reordered = Mul([Tensor("C", ["y", "x"]), Tensor("A", ["x", "z"]), Tensor("B", ["z", "y"])])

    assert canonicalize(left, metadata) == canonicalize(reordered, metadata)


def test_canonicalize_relabels_internal_indices_independently_of_their_names() -> None:
    metadata = Metadata(indices={"occupied": ["i", "j", "k", "l", "m"]}, sizes={"occupied": 8})
    left = Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j", "k"])])
    renamed = Mul([Tensor("A", ["i", "m"]), Tensor("B", ["m", "k"])])

    assert canonicalize(left, metadata) == canonicalize(renamed, metadata)


def test_canonicalize_keeps_index_spaces_distinct() -> None:
    metadata = Metadata(
        indices={"occupied": ["i", "j"], "virtual": ["a", "b"]},
        sizes={"occupied": 8, "virtual": 8},
    )
    occupied = Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j", "i"])])
    virtual = Mul([Tensor("A", ["i", "a"]), Tensor("B", ["a", "i"])])

    assert canonicalize(occupied, metadata) != canonicalize(virtual, metadata)


def test_canonicalize_uses_transitive_tensor_symmetries() -> None:
    metadata = Metadata(
        indices={"all": ["i", "j", "k"]},
        sizes={"all": 8},
        symmetries={"T": [Permutation([1, 2, 0])]},
    )
    first = Tensor("T", ["i", "j", "k"])
    rotated = Tensor("T", ["j", "k", "i"])

    assert canonicalize(first, metadata) == canonicalize(rotated, metadata)


def test_canonicalize_applies_symbol_symmetry_across_space_blocks() -> None:
    metadata = Metadata(
        indices={"occupied": ["i", "j"], "virtual": ["a", "b"]},
        sizes={"occupied": 8, "virtual": 8},
        symmetries={"A": [Permutation([1, 0])]},
    )

    assert canonicalize(Tensor("A", ["i", "j"]), metadata) == canonicalize(Tensor("A", ["j", "i"]), metadata)
    assert canonicalize(Tensor("A", ["a", "b"]), metadata) == canonicalize(Tensor("A", ["b", "a"]), metadata)


def test_canonicalize_preserves_negative_symmetry_phase() -> None:
    metadata = Metadata(
        indices={"all": ["i", "j"]},
        sizes={"all": 8},
        symmetries={"A": [Permutation([1, 0], PhaseFactor.NEGATIVE)]},
    )
    canonical = canonicalize(Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j"])]), metadata)
    swapped = canonicalize(Mul([Tensor("A", ["j", "i"]), Tensor("B", ["j"])]), metadata)

    assert swapped == canonicalize(-canonical, metadata)


def test_canonicalize_zeroes_repeated_axes_of_antisymmetric_tensor() -> None:
    metadata = Metadata(
        indices={"all": ["i"]},
        sizes={"all": 8},
        symmetries={"A": [Permutation([1, 0], PhaseFactor.NEGATIVE)]},
    )

    assert canonicalize(Tensor("A", ["i", "i"]), metadata) == Scalar(0)


def test_canonicalize_handles_symmetry_and_dummy_relabelling_together() -> None:
    metadata = Metadata(
        indices={"all": ["i", "j", "k", "l", "m"]},
        sizes={"all": 8},
        symmetries={"A": [Permutation([1, 0])]},
    )
    left = Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j", "k"])])
    renamed = Mul([Tensor("B", ["m", "l"]), Tensor("A", ["m", "i"])])

    assert canonicalize(left, metadata) == canonicalize(renamed, metadata)


def test_canonicalize_is_idempotent_for_a_sum_of_contracted_networks() -> None:
    metadata = Metadata(indices={"all": ["i", "j", "k", "l", "m", "n"]}, sizes={"all": 8})
    first = Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j", "k"]), Tensor("C", ["k", "i"])])
    second = Mul([Tensor("C", ["m", "l"]), Tensor("A", ["l", "n"]), Tensor("B", ["n", "m"])])
    expression = Add([first, second])

    canonical = canonicalize(expression, metadata)

    assert canonicalize(canonical, metadata) == canonical
    assert canonical_key(canonicalize(first, metadata)) == canonical_key(canonicalize(second, metadata))


def test_canonicalize_handles_large_additions_deterministically() -> None:
    metadata = Metadata(indices={"all": ["i", "j", "k"]}, sizes={"all": 8})
    terms = [Tensor(f"T{number:02}", ["i", "j"]) for number in range(40)]
    forward = Add(terms)
    backward = Add(list(reversed(terms)))

    assert canonicalize(forward, metadata) == canonicalize(backward, metadata)


def test_metadata_rejects_indices_assigned_to_multiple_spaces() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        Metadata(indices={"occupied": ["i"], "virtual": ["i"]}, sizes={"occupied": 8, "virtual": 8})


def test_canonicalize_rejects_symmetry_generators_with_the_wrong_rank() -> None:
    metadata = Metadata(
        indices={"all": ["i", "j", "k"]},
        sizes={"all": 8},
        symmetries={"A": [Permutation([1, 0])]},
    )

    with pytest.raises(ValueError, match="rank mismatch"):
        canonicalize(Tensor("A", ["i", "j", "k"]), metadata)


def test_canonicalize_rejects_inconsistent_symmetry_phases() -> None:
    metadata = Metadata(
        indices={"all": ["i"]},
        sizes={"all": 8},
        symmetries={"A": [Permutation([0], PhaseFactor.NEGATIVE)]},
    )

    with pytest.raises(ValueError, match="inconsistent"):
        canonicalize(Tensor("A", ["i"]), metadata)


def test_canonicalize_rejects_indices_that_occur_more_than_twice() -> None:
    expression = Mul([Tensor("A", ["i"]), Tensor("B", ["i"]), Tensor("C", ["i"])])

    with pytest.raises(ValueError, match="more than twice"):
        canonicalize(expression)
