import pytest

from albert import Add, Expression, Metadata, Mul, Tensor
from albert.tools import diagonal, get_intermediate_output_keys, split_by_dependencies, trace


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


def test_diagonal_multiplies_by_delta_tensors_for_external_index_pairs() -> None:
    node = Tensor("A", ["i", "j", "k"])

    result = diagonal(node, [("i", "j")])

    assert result == Mul([node, Tensor("δ", ["i", "j"])])
    assert result.external_indices == ["k"]


def test_trace_replaces_each_second_index_with_its_first() -> None:
    node = Tensor("A", ["i", "j", "k", "l"])

    result = trace(node, [("i", "j"), ("k", "l")])

    assert result == Tensor("A", ["i", "i", "k", "k"])
    assert result.external_indices == []


@pytest.mark.parametrize("operation", [diagonal, trace])
def test_diagonal_and_trace_require_external_indices(operation) -> None:
    node = Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k", "j"])])

    with pytest.raises(ValueError, match="'k' is not external"):
        operation(node, [("i", "k")])

    with pytest.raises(ValueError, match="'x' is not external"):
        operation(node, [("i", "x")])
