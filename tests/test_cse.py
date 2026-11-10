import pytest

from albert import (
    Add,
    CostModel,
    Expression,
    Metadata,
    Mul,
    PathRequest,
    PathRequests,
    Scalar,
    ScalingLimits,
    Tensor,
    canonical_key,
    canonicalize,
    collect,
    common_subexpression_elimination,
    expand,
    substitute_expressions,
)

CSE_OPTION_SETS = [
    pytest.param(
        {
            "intermediate_format": "__im{}__",
            "max_path_samples": 19,
            "max_bicliques": 8,
            "exhaustive_limit": 4,
        },
        id="defaults",
    ),
    pytest.param(
        {
            "intermediate_format": "__tmp{}__",
            "max_path_samples": 1,
            "max_bicliques": 1,
            "exhaustive_limit": 2,
        },
        id="small-search",
    ),
    pytest.param(
        {
            "intermediate_format": "__shared{}__",
            "max_path_samples": 32,
            "max_bicliques": 16,
            "exhaustive_limit": 5,
        },
        id="wide-search",
    ),
    pytest.param(
        {
            "intermediate_format": "__im{}__",
            "max_path_samples": 19,
            "max_bicliques": 8,
            "exhaustive_limit": 4,
            "scaling_limits": ScalingLimits(),
        },
        id="empty-scaling-limits",
    ),
]


def metadata() -> Metadata:
    return Metadata(
        indices={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"]},
        sizes={"I": 20, "J": 20, "K": 20, "L": 20},
    )


def shared_sum(symbol: str) -> Expression:
    a = Tensor("A", ["i", "k"])
    b = Tensor("B", ["k", "j"])
    c = Tensor("C", ["j", "l"])
    d = Tensor("D", ["j", "l"])
    rhs = Add([Mul([a, b, c]), Mul([a, b, d])])
    return Expression(Tensor(symbol, ["i", "l"]), rhs)


@pytest.mark.parametrize("cse_options", CSE_OPTION_SETS)
def test_cse_factors_a_shared_contraction_and_topologically_sorts(cse_options: dict) -> None:
    result = common_subexpression_elimination(
        [shared_sum("Y")],
        metadata(),
        cost_model=CostModel(multiply=1, add=0, read=0, write=0),
        **cse_options,
    )

    assert len(result) == 3
    positions = {expression.lhs.symbol: i for i, expression in enumerate(result)}
    output = next(expression for expression in result if expression.lhs.symbol == "Y")
    intermediate_symbols = set(positions) - {"Y"}
    output_symbols = {tensor.symbol for tensor in output.rhs.search(lambda node: isinstance(node, Tensor))}
    assert len(intermediate_symbols) == 2
    definitions = {expression.lhs.symbol: expression for expression in result}
    reachable = set(output_symbols)
    pending = list(output_symbols & intermediate_symbols)
    while pending:
        definition = definitions[pending.pop()]
        dependencies = {
            tensor.symbol for tensor in definition.rhs.search(lambda node: isinstance(node, Tensor))
        } & intermediate_symbols
        new_dependencies = dependencies - reachable
        reachable.update(new_dependencies)
        pending.extend(new_dependencies)
    assert intermediate_symbols <= reachable
    assert all(positions[symbol] < positions["Y"] for symbol in intermediate_symbols)

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


@pytest.mark.parametrize("cse_options", CSE_OPTION_SETS)
def test_cse_round_trip_through_expression_substitution(cse_options: dict) -> None:
    inputs = [shared_sum("Y"), shared_sum("Z")]
    factored = common_subexpression_elimination(
        inputs,
        metadata(),
        cost_model=CostModel(multiply=1, add=0, read=0, write=0),
        **cse_options,
    )

    restored = substitute_expressions(factored, metadata())
    expected = {expression.lhs.symbol: expression for expression in inputs}
    actual = {expression.lhs.symbol: expression for expression in restored}

    assert actual.keys() == expected.keys()
    for symbol in expected:
        assert canonical_key(canonicalize(collect(expand(actual[symbol].rhs)), metadata())) == canonical_key(
            canonicalize(collect(expand(expected[symbol].rhs)), metadata())
        )


@pytest.mark.parametrize("cse_options", CSE_OPTION_SETS)
def test_cse_round_trip_maps_indices_across_space_aliases(cse_options: dict) -> None:
    meta = Metadata(
        indices={"I": ["i", "m"], "J": ["j"], "K": ["k", "q"], "L": ["l", "n"]},
        sizes={"I": 20, "J": 20, "K": 20, "L": 20},
    )
    first = shared_sum("Y")
    a = Tensor("A", ["m", "q"])
    b = Tensor("B", ["q", "j"])
    c = Tensor("C", ["j", "n"])
    d = Tensor("D", ["j", "n"])
    second = Expression(Tensor("Z", ["m", "n"]), Add([Mul([a, b, c]), Mul([a, b, d])]))
    inputs = [first, second]

    factored = common_subexpression_elimination(
        inputs,
        meta,
        cost_model=CostModel(multiply=1, add=0, read=0, write=0),
        **cse_options,
    )
    restored = substitute_expressions(factored, meta)
    expected = {expression.lhs.symbol: expression for expression in inputs}
    actual = {expression.lhs.symbol: expression for expression in restored}

    assert actual.keys() == expected.keys()
    for symbol in expected:
        assert canonical_key(canonicalize(collect(expand(actual[symbol].rhs)), meta)) == canonical_key(
            canonicalize(collect(expand(expected[symbol].rhs)), meta)
        )


@pytest.mark.parametrize("cse_options", CSE_OPTION_SETS)
@pytest.mark.parametrize(
    "path_requests",
    [
        PathRequests(
            bounds=(2,),
            requests=(PathRequest(greedy=2), PathRequest(branch=1, greedy=2, dynamic=True)),
        ),
        PathRequests(
            bounds=(2,),
            requests=(PathRequest(greedy=1), PathRequest(branch=1, greedy=1, dynamic=True)),
        ),
    ],
    ids=["greedy-2", "greedy-1"],
)
def test_cse_accepts_path_request_policy(cse_options: dict, path_requests: PathRequests) -> None:
    result = common_subexpression_elimination(
        [shared_sum("Y")],
        metadata(),
        cost_model=CostModel(multiply=1, add=0, read=0, write=0),
        path_requests=path_requests,
        **cse_options,
    )

    assert len(result) == 3


@pytest.mark.parametrize("cse_options", CSE_OPTION_SETS)
def test_cse_shares_generated_intermediates_across_outputs(cse_options: dict) -> None:
    result = common_subexpression_elimination(
        [shared_sum("Y"), shared_sum("Z")],
        metadata(),
        cost_model=CostModel(multiply=1, add=0, read=0, write=0),
        **cse_options,
    )

    assert len(result) == 4
    definitions = {expression.lhs.symbol: expression for expression in result}
    outputs = [definitions["Y"], definitions["Z"]]
    references = [
        {tensor.symbol for tensor in output.rhs.search(lambda node: isinstance(node, Tensor))} for output in outputs
    ]
    assert references[0] == references[1]
    assert len(references[0]) == 2
    assert references[0] & (definitions.keys() - {"Y", "Z"})


@pytest.mark.parametrize("cse_options", CSE_OPTION_SETS)
def test_cse_reuses_intermediates_with_renamed_internal_indices(cse_options: dict) -> None:
    meta = Metadata(
        indices={"I": ["i"], "J": ["j"], "K": ["k", "q"], "L": ["l"]},
        sizes={"I": 20, "J": 20, "K": 20, "L": 20},
    )
    first = shared_sum("Y")
    a = Tensor("A", ["i", "q"])
    b = Tensor("B", ["q", "j"])
    c = Tensor("C", ["j", "l"])
    d = Tensor("D", ["j", "l"])
    second = Expression(Tensor("Z", ["i", "l"]), Add([Mul([a, b, c]), Mul([a, b, d])]))

    result = common_subexpression_elimination([first, second], meta, **cse_options)
    intermediate_keys = [
        canonical_key(canonicalize(expression.rhs, meta))
        for expression in result
        if expression.lhs.symbol not in {"Y", "Z"}
    ]

    assert len(result) == 4
    assert len(intermediate_keys) == len(set(intermediate_keys))


@pytest.mark.parametrize("cse_options", CSE_OPTION_SETS)
def test_cse_advances_past_multiple_reserved_intermediate_names(cse_options: dict) -> None:
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

    result = common_subexpression_elimination([expression], metadata(), **cse_options)
    symbols = [item.lhs.symbol for item in result]

    assert len(result) == 3
    assert len(symbols) == len(set(symbols))


@pytest.mark.parametrize("cse_options", CSE_OPTION_SETS)
def test_cse_leaves_unprofitable_expressions_unchanged(cse_options: dict) -> None:
    expression = Expression(
        Tensor("Y", ["i", "l"]),
        Mul([Scalar(2), Tensor("A", ["i", "k"]), Tensor("B", ["k", "l"])]),
    )

    result = common_subexpression_elimination([expression], metadata(), **cse_options)

    assert len(result) == 1
    assert result[0].lhs.symbol == "Y"
    assert canonical_key(canonicalize(collect(expand(result[0].rhs)), metadata())) == canonical_key(
        canonicalize(collect(expand(expression.rhs)), metadata())
    )


@pytest.mark.parametrize("cse_options", CSE_OPTION_SETS)
def test_cse_cost_factors_are_opt_in_and_configurable(cse_options: dict) -> None:
    expression = shared_sum("Y")
    meta = Metadata(
        indices={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"]},
        sizes={"I": 20, "J": 20, "K": 20, "L": 20},
        cost_factors={"A": 2, "B": 3, "C": 5, "D": 7},
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
        **cse_options,
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
        **cse_options,
    )

    assert len(product) == len(maximum)
    assert len(product) > 1
