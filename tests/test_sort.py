import pytest

from albert import Add, Expression, Mul, Tensor, topological_sort


def _expression(symbol: str, rhs) -> Expression:
    return Expression(Tensor(symbol, rhs.external_indices), rhs)


def _symbols(expressions: list[Expression]) -> list[str]:
    return [expression.lhs.symbol for expression in expressions]


def test_topological_sort_uses_input_order_for_ready_expressions() -> None:
    expressions = [
        _expression("R", Tensor("B", ["i"])),
        _expression("free_1", Tensor("X", ["i"])),
        _expression("free_2", Tensor("Y", ["i"])),
    ]

    assert _symbols(topological_sort(expressions)) == ["R", "free_1", "free_2"]


def test_topological_sort_selects_earliest_currently_ready_expression() -> None:
    output = _expression("R", Tensor("intermediate", ["i"]))
    independent = _expression("free", Tensor("input", ["i"]))
    intermediate = _expression("intermediate", Tensor("A", ["i"]))

    ordered = topological_sort([output, independent, intermediate])

    assert _symbols(ordered) == ["free", "intermediate", "R"]


def test_topological_sort_orders_chains_and_branching_dependencies() -> None:
    source = _expression("source", Tensor("A", ["i"]))
    left = _expression("left", Mul([Tensor("source", ["i"]), Tensor("B", ["i"])]))
    right = _expression("right", Mul([Tensor("source", ["i"]), Tensor("C", ["i"])]))
    joined = _expression("joined", Add([Tensor("left", ["i"]), Tensor("right", ["i"])]))
    independent = _expression("independent", Tensor("D", ["i"]))

    ordered = topological_sort([joined, right, independent, left, source])
    symbols = _symbols(ordered)

    assert symbols == ["independent", "source", "right", "left", "joined"]


def test_topological_sort_deduplicates_repeated_references() -> None:
    source = _expression("source", Tensor("A", ["i"]))
    uses_source_twice = _expression(
        "uses_source_twice",
        Add([Tensor("source", ["i"]), Tensor("source", ["i"])]),
    )

    assert _symbols(topological_sort([uses_source_twice, source])) == [
        "source",
        "uses_source_twice",
    ]


def test_topological_sort_ignores_undefined_input_symbols_and_accepts_empty() -> None:
    output = _expression("R", Tensor("input", ["i"]))

    assert _symbols(topological_sort([output])) == ["R"]
    assert topological_sort([]) == []


def test_topological_sort_rejects_duplicate_definitions() -> None:
    first = _expression("tmp", Tensor("A", ["i"]))
    second = _expression("tmp", Tensor("B", ["i"]))

    with pytest.raises(ValueError, match="duplicate expression definition"):
        topological_sort([first, second])


@pytest.mark.parametrize(
    "expressions",
    [
        [
            _expression("first", Tensor("second", ["i"])),
            _expression("second", Tensor("first", ["i"])),
        ],
        [_expression("self", Tensor("self", ["i"]))],
    ],
)
def test_topological_sort_rejects_cycles(expressions: list[Expression]) -> None:
    with pytest.raises(ValueError, match="cycle detected"):
        topological_sort(expressions)


def test_topological_sort_detects_cycle_even_with_ready_expressions() -> None:
    independent = _expression("independent", Tensor("A", ["i"]))
    first = _expression("first", Tensor("second", ["i"]))
    second = _expression("second", Tensor("first", ["i"]))

    with pytest.raises(ValueError, match="cycle detected"):
        topological_sort([independent, first, second])


def test_topological_sort_handles_a_long_chain_with_many_terms() -> None:
    count = 80
    definitions = []
    for i in reversed(range(count)):
        if i == 0:
            rhs = Add([Tensor(f"input_{j}", []) for j in range(8)])
        else:
            rhs = Add([Tensor(f"tmp_{i - 1}", []) for _ in range(4)] + [Tensor(f"input_{i}_{j}", []) for j in range(8)])
        definitions.append(_expression(f"tmp_{i}", rhs))

    ordered = topological_sort(definitions)

    assert _symbols(ordered) == [f"tmp_{i}" for i in range(count)]


def test_topological_sort_handles_wide_fanin_and_many_independent_definitions() -> None:
    symbols = [f"leaf_{i:03}" for i in range(80)]
    leaves = [_expression(symbol, Tensor(f"input_{symbol}", [])) for symbol in symbols]
    output = _expression("output", Add([Tensor(symbol, []) for symbol in symbols]))

    ordered = topological_sort([output, *reversed(leaves)])

    assert _symbols(ordered) == [*reversed(symbols), "output"]
