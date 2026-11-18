import pytest

from albert import Add, Expression, Metadata, Mul, Permutation, PhaseFactor, Scalar, Tensor
from albert.tools import diagonal, get_intermediate_output_keys, operator_diagonal, split_by_dependencies, trace


def _symbols(expressions: list[Expression]) -> list[str]:
    return [expression.lhs.symbol for expression in expressions]


def test_split_by_dependencies_propagates_through_intermediates() -> None:
    metadata = Metadata(indices={"o": ["i", "j"]}, sizes={"o": 2})
    static = Expression(Tensor("static", ["i"]), Tensor("A", ["i"]))
    first = Expression(
        Tensor("first", ["i"]),
        Mul([Tensor("t", ["i", "j"]), Tensor("B", ["j"])]),
    )
    second = Expression(Tensor("second", ["i"]), Tensor("first", ["i"]))
    result = Expression(Tensor("result", ["i"]), Add([Tensor("static", ["i"]), Tensor("second", ["i"])]))

    independent, dependent = split_by_dependencies([result, second, first, static], {"t"}, metadata=metadata)

    assert _symbols(independent) == ["static"]
    assert _symbols(dependent) == ["first", "second", "result"]


def test_split_by_dependencies_matches_intermediates_by_symbol_and_spaces() -> None:
    metadata = Metadata(indices={"o": ["i", "j"], "v": ["a"]}, sizes={"o": 2, "v": 3})
    occupied = Expression(
        Tensor("P", ["i"]),
        Mul([Tensor("t", ["i", "j"]), Tensor("B", ["j"])]),
    )
    virtual = Expression(Tensor("P", ["a"]), Tensor("A", ["a"]))
    occupied_result = Expression(Tensor("R", ["i"]), Tensor("P", ["i"]))
    virtual_result = Expression(Tensor("S", ["a"]), Tensor("P", ["a"]))

    independent, dependent = split_by_dependencies(
        [occupied_result, virtual_result, virtual, occupied], {"t"}, metadata=metadata
    )

    assert [(expression.lhs.symbol, tuple(expression.lhs.indices)) for expression in independent] == [
        ("P", ("a",)),
        ("S", ("a",)),
    ]
    assert [(expression.lhs.symbol, tuple(expression.lhs.indices)) for expression in dependent] == [
        ("P", ("i",)),
        ("R", ("i",)),
    ]


def test_split_by_dependencies_classifies_dependency_symbol_definitions_as_dependent() -> None:
    metadata = Metadata(indices={"o": ["i"]}, sizes={"o": 2})
    amplitude_update = Expression(Tensor("t", ["i"]), Tensor("A", ["i"]))
    independent = Expression(Tensor("static", ["i"]), Tensor("B", ["i"]))

    independent_part, dependent_part = split_by_dependencies([independent, amplitude_update], {"t"}, metadata=metadata)

    assert _symbols(independent_part) == ["static"]
    assert _symbols(dependent_part) == ["t"]


def test_get_intermediate_output_keys_matches_symbol_and_spaces_in_producer_order() -> None:
    metadata = Metadata(indices={"o": ["i", "j"], "v": ["a"]}, sizes={"o": 2, "v": 3})
    occupied = Expression(Tensor("P", ["i"]), Tensor("A", ["i"]))
    virtual = Expression(Tensor("P", ["a"]), Tensor("B", ["a"]))
    consumer = Expression(
        Tensor("result", ["i", "a"]),
        Mul([Tensor("P", ["i"]), Tensor("P", ["a"])]),
    )

    outputs = get_intermediate_output_keys([virtual, occupied], [consumer], metadata=metadata)

    assert outputs == (("P", ("v",)), ("P", ("o",)))


def test_diagonal_uses_three_index_equality_tensor_for_external_index_pairs() -> None:
    metadata = Metadata(indices={"o": ["i", "j", "k", "l"]}, sizes={"o": 4})
    node = Tensor("A", ["i", "j"])

    result = diagonal(node, [("i", "j")], metadata=metadata)

    assert result == Mul([Tensor("A", ["j", "k"]), Tensor("δ", ["i", "j", "k"])])
    assert result.external_indices == ["i"]


def test_diagonal_keeps_unpaired_external_indices_and_supports_multiple_pairs() -> None:
    metadata = Metadata(indices={"o": ["i", "j", "k", "l", "m", "n"]}, sizes={"o": 6})
    node = Tensor("A", ["i", "j", "k", "l"])

    result = diagonal(node, [("i", "j"), ("k", "l")], metadata=metadata)

    assert result == Mul(
        [
            Tensor("A", ["j", "m", "l", "n"]),
            Tensor("δ", ["i", "j", "m"]),
            Tensor("δ", ["k", "l", "n"]),
        ]
    )
    assert result.external_indices == ["i", "k"]


def test_diagonal_requires_same_space_and_a_fresh_dummy_index() -> None:
    metadata = Metadata(indices={"o": ["i", "j"], "v": ["a", "b"]}, sizes={"o": 2, "v": 2})
    node = Tensor("A", ["i", "a"])

    with pytest.raises(ValueError, match="must belong to the same space"):
        diagonal(node, [("i", "a")], metadata=metadata)

    no_free_index_metadata = Metadata(indices={"o": ["i", "j"]}, sizes={"o": 2})
    with pytest.raises(ValueError, match="no unused index is available"):
        diagonal(Tensor("A", ["i", "j"]), [("i", "j")], metadata=no_free_index_metadata)


def test_diagonal_requires_disjoint_index_pairs() -> None:
    metadata = Metadata(indices={"o": ["i", "j", "k", "l", "m"]}, sizes={"o": 5})
    node = Tensor("A", ["i", "j", "k"])

    with pytest.raises(ValueError, match="index pairs must be disjoint"):
        diagonal(node, [("i", "j"), ("i", "k")], metadata=metadata)


def test_operator_diagonal_eliminates_the_trial_vector() -> None:
    metadata = Metadata(indices={"o": ["i", "j", "k", "l"]}, sizes={"o": 4})
    expression = Expression(
        Tensor("R", ["i"]),
        Mul([Tensor("A", ["i", "j"]), Tensor("r", ["j"])]),
    )

    result = operator_diagonal(expression, ["r"], metadata=metadata)

    assert result == Mul([Tensor("A", ["k", "j"]), Tensor("δ", ["i", "k", "j"])])
    assert result.external_indices == ["i"]


def test_operator_diagonal_of_identity_is_a_ones_tensor() -> None:
    metadata = Metadata(indices={"o": ["i", "j", "k"]}, sizes={"o": 3})
    vector = Tensor("r", ["i"])

    result = operator_diagonal(Expression(vector, vector), ["r"], metadata=metadata)

    assert result == Tensor("δ", ["i", "j", "j"])
    assert result.external_indices == ["i"]


def test_operator_diagonal_preserves_external_indices_shared_with_vector() -> None:
    metadata = Metadata(indices={"o": ["i", "j", "k", "l", "m", "n"]}, sizes={"o": 6})
    expression = Expression(
        Tensor("R", ["i", "j"]),
        Mul([Tensor("A", ["i", "k"]), Tensor("r", ["j", "k"])]),
    )

    result = operator_diagonal(expression, ["r"], metadata=metadata)

    assert result.external_indices == ["i", "j"]


def test_operator_diagonal_accounts_for_vector_antisymmetry() -> None:
    metadata = Metadata(
        indices={"o": ["i", "j", "k", "l", "m", "n", "p", "q"]},
        sizes={"o": 6},
        symmetries={"r": [Permutation([1, 0], PhaseFactor.NEGATIVE)]},
    )
    expression = Expression(
        Tensor("r", ["i", "j"]),
        Mul([Tensor("A", ["i", "j", "k", "l"]), Tensor("r", ["k", "l"])]),
    )

    result = operator_diagonal(expression, ["r"], metadata=metadata)

    assert isinstance(result, Add)
    assert len(result.children) == 2
    assert result.external_indices == ["i", "j"]
    assert any(
        child == Scalar(-1.0)
        for term in result.children
        for child in term.search(lambda node: isinstance(node, Scalar))
    )


def test_operator_diagonal_ignores_other_vector_blocks_and_rejects_nonlinear_terms() -> None:
    metadata = Metadata(
        indices={"o": ["i", "j", "k"], "v": ["a", "b"]},
        sizes={"o": 3, "v": 2},
    )
    off_diagonal = Expression(
        Tensor("r1", ["i"]),
        Mul([Tensor("r2", ["i", "a"]), Tensor("t", ["a"])]),
    )
    assert operator_diagonal(off_diagonal, ["r1", "r2"], metadata=metadata) == Scalar(0.0)

    nonlinear = Expression(
        Tensor("r1", ["i"]),
        Mul([Tensor("A", ["i", "j", "k"]), Tensor("r1", ["j"]), Tensor("r2", ["k"])]),
    )
    with pytest.raises(ValueError, match="linear in its vector symbols"):
        operator_diagonal(nonlinear, ["r1", "r2"], metadata=metadata)


def test_trace_replaces_each_second_index_with_its_first() -> None:
    node = Tensor("A", ["i", "j", "k", "l"])

    result = trace(node, [("i", "j"), ("k", "l")])

    assert result == Tensor("A", ["i", "i", "k", "k"])
    assert result.external_indices == []


def test_diagonal_and_trace_require_external_indices() -> None:
    metadata = Metadata(indices={"o": ["i", "j", "k", "x"]}, sizes={"o": 4})
    node = Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k", "j"])])

    with pytest.raises(ValueError, match="'k' is not external"):
        diagonal(node, [("i", "k")], metadata=metadata)

    with pytest.raises(ValueError, match="'x' is not external"):
        diagonal(node, [("i", "x")], metadata=metadata)

    with pytest.raises(ValueError, match="'k' is not external"):
        trace(node, [("i", "k")])

    with pytest.raises(ValueError, match="'x' is not external"):
        trace(node, [("i", "x")])
