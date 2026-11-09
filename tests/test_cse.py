from albert import (
    Add,
    CostModel,
    Expression,
    Metadata,
    Mul,
    PathRequest,
    PathRequests,
    Scalar,
    Tensor,
    canonical_key,
    canonicalize,
    collect,
    common_subexpression_elimination,
    expand,
    substitute_expressions,
)


def metadata() -> Metadata:
    return Metadata(
        spaces={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"]},
        sizes={"I": 20, "J": 20, "K": 20, "L": 20},
    )


def shared_sum(symbol: str) -> Expression:
    a = Tensor("A", ["i", "k"])
    b = Tensor("B", ["k", "j"])
    c = Tensor("C", ["j", "l"])
    d = Tensor("D", ["j", "l"])
    rhs = Add([Mul([a, b, c]), Mul([a, b, d])])
    return Expression(Tensor(symbol, ["i", "l"]), rhs)


def test_cse_factors_a_shared_contraction_and_topologically_sorts() -> None:
    result = common_subexpression_elimination(
        [shared_sum("Y")], metadata(), cost_model=CostModel(multiply=1, add=0, read=0, write=0)
    )

    assert len(result) == 3
    positions = {expression.lhs.symbol: i for i, expression in enumerate(result)}
    output = next(expression for expression in result if expression.lhs.symbol == "Y")
    assert output.rhs.search(lambda node: isinstance(node, Tensor))[0].symbol.startswith("__im")
    assert positions[output.rhs.search(lambda node: isinstance(node, Tensor))[0].symbol] < positions["Y"]
    assert positions[output.rhs.search(lambda node: isinstance(node, Tensor))[1].symbol] < positions["Y"]

    definitions = {expression.lhs.symbol: expression for expression in result}

    def inline(node):
        def replace(item):
            if not isinstance(item, Tensor) or item.symbol not in definitions:
                return item
            definition = definitions[item.symbol]
            expanded = inline(definition.rhs)
            return expanded.map_indices(dict(zip(definition.lhs.indices, item.indices)))

        return node.apply(replace, predicate=lambda item: isinstance(item, Tensor))

    expected = collect(expand(shared_sum("Y").rhs))
    actual = collect(expand(inline(output.rhs)))
    assert canonical_key(canonicalize(actual, metadata())) == canonical_key(canonicalize(expected, metadata()))


def test_cse_round_trip_through_expression_substitution() -> None:
    inputs = [shared_sum("Y"), shared_sum("Z")]
    factored = common_subexpression_elimination(
        inputs, metadata(), cost_model=CostModel(multiply=1, add=0, read=0, write=0)
    )

    restored = substitute_expressions(factored)
    expected = {expression.lhs.symbol: expression for expression in inputs}
    actual = {expression.lhs.symbol: expression for expression in restored}

    assert actual.keys() == expected.keys()
    for symbol in expected:
        assert canonical_key(canonicalize(collect(expand(actual[symbol].rhs)), metadata())) == canonical_key(
            canonicalize(collect(expand(expected[symbol].rhs)), metadata())
        )


def test_cse_accepts_path_request_policy() -> None:
    result = common_subexpression_elimination(
        [shared_sum("Y")],
        metadata(),
        cost_model=CostModel(multiply=1, add=0, read=0, write=0),
        path_requests=PathRequests(
            bounds=(2,),
            requests=(PathRequest(greedy=2), PathRequest(branch=1, greedy=2, dynamic=True)),
        ),
    )

    assert len(result) == 3


def test_cse_shares_generated_intermediates_across_outputs() -> None:
    result = common_subexpression_elimination(
        [shared_sum("Y"), shared_sum("Z")],
        metadata(),
        cost_model=CostModel(multiply=1, add=0, read=0, write=0),
    )

    assert len(result) == 4
    definitions = {expression.lhs.symbol: expression for expression in result}
    outputs = [definitions["Y"], definitions["Z"]]
    references = [
        {tensor.symbol for tensor in output.rhs.search(lambda node: isinstance(node, Tensor))} for output in outputs
    ]
    assert references[0] == references[1]
    assert len(references[0]) == 2
    assert all(symbol in definitions for symbol in references[0])


def test_cse_reuses_intermediates_with_renamed_internal_indices() -> None:
    meta = Metadata(
        spaces={"I": ["i"], "J": ["j"], "K": ["k", "q"], "L": ["l"]},
        sizes={"I": 20, "J": 20, "K": 20, "L": 20},
    )
    first = shared_sum("Y")
    a = Tensor("A", ["i", "q"])
    b = Tensor("B", ["q", "j"])
    c = Tensor("C", ["j", "l"])
    d = Tensor("D", ["j", "l"])
    second = Expression(Tensor("Z", ["i", "l"]), Add([Mul([a, b, c]), Mul([a, b, d])]))

    result = common_subexpression_elimination([first, second], meta)
    intermediate_keys = [
        canonical_key(canonicalize(expression.rhs, meta))
        for expression in result
        if expression.lhs.symbol not in {"Y", "Z"}
    ]

    assert len(result) == 4
    assert len(intermediate_keys) == len(set(intermediate_keys))


def test_cse_advances_past_multiple_reserved_intermediate_names() -> None:
    expression = Expression(
        Tensor("Y", ["i", "l"]),
        Add(
            [
                Mul(
                    [
                        Tensor("__im5__", ["i", "k"]),
                        Tensor("__im6__", ["k", "j"]),
                        Tensor("C", ["j", "l"]),
                    ]
                ),
                Mul(
                    [
                        Tensor("__im5__", ["i", "k"]),
                        Tensor("__im6__", ["k", "j"]),
                        Tensor("D", ["j", "l"]),
                    ]
                ),
            ]
        ),
    )

    result = common_subexpression_elimination([expression], metadata())
    symbols = [item.lhs.symbol for item in result]

    assert len(result) == 3
    assert len(symbols) == len(set(symbols))


def test_cse_leaves_unprofitable_expressions_unchanged() -> None:
    expression = Expression(
        Tensor("Y", ["i", "l"]),
        Mul([Scalar(2), Tensor("A", ["i", "k"]), Tensor("B", ["k", "l"])]),
    )

    result = common_subexpression_elimination([expression], metadata())

    assert len(result) == 1
    assert result[0].lhs.symbol == "Y"
    assert canonical_key(canonicalize(collect(expand(result[0].rhs)), metadata())) == canonical_key(
        canonicalize(collect(expand(expression.rhs)), metadata())
    )


def test_cse_cost_factors_are_opt_in_and_configurable() -> None:
    expression = shared_sum("Y")
    meta = Metadata(
        spaces={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"]},
        sizes={"I": 20, "J": 20, "K": 20, "L": 20},
        cost_factor={"A": 2, "B": 3, "C": 5, "D": 7},
    )

    product = common_subexpression_elimination(
        [expression],
        meta,
        cost_model=CostModel(
            multiply=1,
            add=0,
            read=0,
            write=0,
            apply_cost_factors=True,
            cost_factor_aggregation="product",
        ),
    )
    maximum = common_subexpression_elimination(
        [expression],
        meta,
        cost_model=CostModel(
            multiply=1,
            add=0,
            read=0,
            write=0,
            apply_cost_factors=True,
            cost_factor_aggregation="maximum",
        ),
    )

    assert len(product) == len(maximum)
    assert len(product) > 1
