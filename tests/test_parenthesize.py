import pytest

from albert import (
    CostModel,
    Metadata,
    Mul,
    PathRequest,
    PathRequests,
    ScalingLimits,
    ScalingMetric,
    Tensor,
    apply_parenthesization,
    parenthesization_candidates,
    scaling_exponent,
)


@pytest.fixture
def chain():
    a = Tensor("A", ["i", "j"])
    b = Tensor("B", ["j", "k"])
    c = Tensor("C", ["k", "l"])
    metadata = Metadata(indices={"n": ["i", "j", "k", "l"]}, sizes={"n": 10})
    return Mul([a, b, c]), metadata, a, b, c


def test_exhaustive_parenthesization_returns_pairwise_paths(chain) -> None:
    node, metadata, *_ = chain

    candidates = parenthesization_candidates(node, metadata, max_candidates=10, exhaustive_limit=3)

    assert len(candidates) == 3
    assert all(len(candidate.path) == 2 for candidate in candidates)
    assert all(all(len(pair) == 2 for pair in candidate.path) for candidate in candidates)
    assert all(candidate.cost.total >= 0 for candidate in candidates)


def test_parenthesization_candidates_are_ranked_and_capped(chain) -> None:
    node, metadata, *_ = chain

    candidates = parenthesization_candidates(node, metadata, max_candidates=2, exhaustive_limit=3)

    assert len(candidates) == 2
    assert [candidate.cost.total for candidate in candidates] == sorted(
        candidate.cost.total for candidate in candidates
    )


def test_path_requests_select_algorithms_by_operand_count(chain) -> None:
    node, metadata, *_ = chain
    requests = PathRequests(
        bounds=(2,),
        requests=(PathRequest(greedy=2), PathRequest(branch=2, greedy=3)),
    )

    assert requests.for_size(2) == PathRequest(greedy=2)
    assert requests.for_size(3) == PathRequest(branch=2, greedy=3)
    assert PathRequests.default() == PathRequests(
        bounds=(4,),
        requests=(PathRequest(exhaustive=18, dynamic=True), PathRequest(dynamic=True)),
    )
    with pytest.raises(ValueError, match="one more item"):
        PathRequests(bounds=(2,), requests=(PathRequest(),))


@pytest.mark.parametrize(
    "path_request",
    [PathRequest(branch=1), PathRequest(greedy=4), PathRequest(branch=2, greedy=4)],
)
def test_bounded_searches_return_valid_ranked_candidates(chain, path_request) -> None:
    node, metadata, *_ = chain
    policy = PathRequests(bounds=(), requests=(path_request,))

    candidates = parenthesization_candidates(node, metadata, max_candidates=8, path_requests=policy)

    assert candidates
    assert all(len(candidate.path) == node.arity - 1 for candidate in candidates)
    assert [candidate.cost.total for candidate in candidates] == sorted(
        candidate.cost.total for candidate in candidates
    )


def test_exhaustive_finds_the_lowest_cost_parenthesization() -> None:
    a = Tensor("A", ["i", "j"])
    b = Tensor("B", ["j", "k"])
    c = Tensor("C", ["k", "l"])
    node = Mul([a, b, c])
    metadata = Metadata(
        indices={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"]},
        sizes={"I": 2, "J": 2, "K": 10_000, "L": 1_000},
    )
    model = CostModel(multiply=1, add=0, read=0, write=0)

    candidates = parenthesization_candidates(node, metadata, max_candidates=10, exhaustive_limit=3, cost_model=model)

    assert len(candidates) == 3
    assert candidates[0].path == [(1, 2), (0, 1)]
    assert candidates[0].cost.total == min(candidate.cost.total for candidate in candidates)
    assert candidates[0].cost.total < candidates[1].cost.total


def test_parenthesization_reads_tensor_cost_factors_from_cost_model(chain) -> None:
    node, metadata, *_ = chain
    model = CostModel(
        multiply=1,
        add=0,
        read=0,
        write=0,
        cost_factors={"A": 100},
        apply_cost_factors=True,
    )

    candidates = parenthesization_candidates(node, metadata, max_candidates=3, exhaustive_limit=3, cost_model=model)
    unweighted = parenthesization_candidates(
        node,
        metadata,
        max_candidates=3,
        exhaustive_limit=3,
        cost_model=CostModel(multiply=1, add=0, read=0, write=0),
    )

    assert candidates[0].path == unweighted[0].path
    assert candidates[0].cost.total == pytest.approx(unweighted[0].cost.total * 100)


def test_parallel_exhaustive_returns_stable_top_candidates() -> None:
    indices = [f"i{index}" for index in range(5)]
    factors = [Tensor(f"T{index}", ["shared", index]) for index in indices]
    node = Mul(factors)
    metadata = Metadata(indices={"n": ["shared", *indices]}, sizes={"n": 2})

    first = parenthesization_candidates(node, metadata, max_candidates=5, exhaustive_limit=5)
    second = parenthesization_candidates(node, metadata, max_candidates=5, exhaustive_limit=5)

    assert len(first) == 5
    assert [(candidate.path, candidate.cost.total) for candidate in first] == [
        (candidate.path, candidate.cost.total) for candidate in second
    ]


def test_parallel_bitmask_dp_returns_stable_path() -> None:
    indices = [f"i{index}" for index in range(10)]
    factors = [Tensor(f"T{index}", indices[index : index + 2]) for index in range(9)]
    node = Mul(factors)
    metadata = Metadata(indices={"n": indices}, sizes={"n": 2})
    requests = PathRequests(bounds=(), requests=(PathRequest(dynamic=True),))

    first = parenthesization_candidates(node, metadata, path_requests=requests)
    second = parenthesization_candidates(node, metadata, path_requests=requests)

    assert len(first) == 1
    assert len(first[0].path) == len(factors) - 1
    assert (first[0].path, first[0].cost.total) == (second[0].path, second[0].cost.total)


def test_parallel_generic_dp_returns_stable_path() -> None:
    factors = []
    for index in range(9):
        shared = []
        if index > 0:
            shared.append(f"c{index - 1}")
        if index < 8:
            shared.append(f"c{index}")
        unique = [f"u{index}_{axis}" for axis in range(7)]
        factors.append(Tensor(f"T{index}", [*shared, *unique]))
    node = Mul(factors)
    axes = sorted({axis for factor in factors for axis in factor.indices})
    metadata = Metadata(indices={"n": axes}, sizes={"n": 1})
    requests = PathRequests(bounds=(), requests=(PathRequest(dynamic=True),))

    first = parenthesization_candidates(node, metadata, path_requests=requests)
    second = parenthesization_candidates(node, metadata, path_requests=requests)

    assert len(first) == 1
    assert len(first[0].path) == len(factors) - 1
    assert (first[0].path, first[0].cost.total) == (second[0].path, second[0].cost.total)


def test_sparse_dp_finds_exact_path_for_more_than_20_connected_operands() -> None:
    indices = [f"i{index}" for index in range(25)]
    factors = [Tensor(f"T{index}", indices[index : index + 2]) for index in range(24)]
    node = Mul(factors)
    metadata = Metadata(indices={"n": indices}, sizes={"n": 2})
    limits = ScalingLimits(time={"n": 3}, memory={"n": 2})
    requests = PathRequests(bounds=(), requests=(PathRequest(dynamic=True),))

    candidates = parenthesization_candidates(
        node,
        metadata,
        scaling_limits=limits,
        path_requests=requests,
    )

    assert len(candidates) == 1
    assert len(candidates[0].path) == len(factors) - 1


def test_sparse_dp_matches_matrix_chain_optimum() -> None:
    dimensions = [3, 7, 2, 11, 5, 13, 4, 17, 6, 19, 8, 23, 9, 29, 10, 31, 12, 37, 14, 41, 16, 43]
    indices = [f"i{index}" for index in range(len(dimensions))]
    factors = [Tensor(f"T{index}", indices[index : index + 2]) for index in range(len(dimensions) - 1)]
    spaces = {f"S{index}": [index_name] for index, index_name in enumerate(indices)}
    sizes = {f"S{index}": size for index, size in enumerate(dimensions)}
    metadata = Metadata(indices=spaces, sizes=sizes)
    model = CostModel(multiply=1, add=0, read=0, write=0)
    requests = PathRequests(bounds=(), requests=(PathRequest(dynamic=True),))

    expected = [[0.0] * len(factors) for _ in factors]
    for width in range(2, len(factors) + 1):
        for start in range(len(factors) - width + 1):
            stop = start + width - 1
            expected[start][stop] = min(
                expected[start][split]
                + expected[split + 1][stop]
                + dimensions[start] * dimensions[split + 1] * dimensions[stop + 1]
                for split in range(start, stop)
            )

    candidates = parenthesization_candidates(
        Mul(factors),
        metadata,
        cost_model=model,
        path_requests=requests,
    )

    assert len(candidates) == 1
    assert candidates[0].cost.total == expected[0][-1]


def test_dp_checks_every_partition_against_exhaustive_search() -> None:
    factors = [
        Tensor("A", ["i", "j"]),
        Tensor("B", ["j", "k"]),
        Tensor("C", ["k", "l"]),
        Tensor("D", ["l", "m"]),
    ]
    node = Mul(factors)
    metadata = Metadata(
        indices={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"], "M": ["m"]},
        sizes={"I": 2, "J": 3, "K": 5, "L": 7, "M": 11},
    )
    model = CostModel(multiply=1, add=0, read=0, write=0)

    exhaustive = parenthesization_candidates(node, metadata, max_candidates=100, exhaustive_limit=4, cost_model=model)
    dynamic = parenthesization_candidates(node, metadata, exhaustive_limit=3, cost_model=model)

    assert len(dynamic) == 1
    assert dynamic[0].cost.total == min(candidate.cost.total for candidate in exhaustive)


def test_dp_bitmask_handles_composite_operands_via_external_indices() -> None:
    composite = Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k", "j"])])
    factors = [composite, Tensor("C", ["j", "l"]), Tensor("D", ["l", "m"])]
    node = Mul(factors)
    metadata = Metadata(
        indices={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"], "M": ["m"]},
        sizes={"I": 2, "J": 3, "K": 5, "L": 7, "M": 11},
    )
    model = CostModel(multiply=1, add=0, read=0, write=0)

    exhaustive = parenthesization_candidates(node, metadata, max_candidates=100, exhaustive_limit=3, cost_model=model)
    dynamic = parenthesization_candidates(node, metadata, exhaustive_limit=2, cost_model=model)

    assert len(dynamic) == 1
    assert dynamic[0].cost.total == min(candidate.cost.total for candidate in exhaustive)


def test_apply_parenthesization_builds_binary_product_tree(chain) -> None:
    node, metadata, a, b, c = chain
    candidate = parenthesization_candidates(node, metadata, exhaustive_limit=3)[0]

    result = apply_parenthesization(node, candidate)

    assert isinstance(result, Mul)
    assert len(result.children) == 2
    products = result.search(lambda node: isinstance(node, Mul))
    assert len(products) == 2
    assert all(len(product.children) == 2 for product in products)
    assert set(result.external_indices) == {"i", "l"}
    assert {a.symbol, b.symbol, c.symbol} == {
        tensor.symbol for tensor in result.search(lambda n: isinstance(n, Tensor))
    }


def test_dp_returns_one_path_even_when_max_candidates_is_zero(chain) -> None:
    node, metadata, *_ = chain

    candidates = parenthesization_candidates(node, metadata, max_candidates=0, exhaustive_limit=2)

    assert len(candidates) == 1
    assert len(candidates[0].path) == 2


def test_dp_allows_outer_product_to_join_disconnected_components() -> None:
    a = Tensor("A", ["i", "j"])
    b = Tensor("B", ["j", "k"])
    c = Tensor("C", ["a", "b"])
    node = Mul([a, b, c])
    metadata = Metadata(
        indices={"n": ["i", "j", "k", "a", "b"]},
        sizes={"n": 2},
    )

    candidates = parenthesization_candidates(node, metadata, exhaustive_limit=2)

    assert len(candidates) == 1
    assert candidates[0].path == [(0, 1), (0, 1)]
    result = apply_parenthesization(node, candidates[0])
    assert set(result.external_indices) == {"i", "k", "a", "b"}


def test_dp_path_uses_opt_einsum_append_order() -> None:
    factors = [
        Tensor("f", ["i", "a"]),
        Tensor("g1", ["r0", "i"]),
        Tensor("g3", ["s0", "a"]),
        Tensor("g2", ["r0", "y0", "s0"]),
    ]
    node = Mul(factors)
    metadata = Metadata(
        indices={
            "o": ["i"],
            "v": ["a"],
            "r": ["r0"],
            "s": ["s0"],
            "y": ["y0"],
            "x": ["x0"],
        },
        sizes={"o": 100, "v": 900, "r": 25, "s": 25, "y": 25, "x": 2000},
    )
    limits = ScalingLimits(
        time={("o", "v", "x"): 3, ("o", "v", "x", "y", "r", "s"): 6},
        memory={("o", "v", "x"): 3, ("o", "v", "x", "y", "r", "s"): 8},
    )

    candidate = parenthesization_candidates(node, metadata, scaling_limits=limits, exhaustive_limit=3)[0]
    result = apply_parenthesization(node, candidate)

    groups = {
        frozenset(tensor.symbol for tensor in child.search(lambda n: isinstance(n, Tensor)))
        for child in result.children
    }
    assert groups == {frozenset({"g2"}), frozenset({"f", "g1", "g3"})}


def test_scaling_caps_filter_exhaustive_paths(chain) -> None:
    node, metadata, *_ = chain
    limits = ScalingLimits(time={"n": 3}, memory={"n": 2})

    candidates = parenthesization_candidates(node, metadata, scaling_limits=limits, exhaustive_limit=3)

    assert len(candidates) == 2
    assert all(candidate.path[0] in {(0, 1), (1, 2)} for candidate in candidates)
    assert all(
        scaling_exponent(apply_parenthesization(node, candidate), "n", ScalingMetric.TIME, metadata) <= 3
        for candidate in candidates
    )
    assert all(
        scaling_exponent(apply_parenthesization(node, candidate), "n", ScalingMetric.MEMORY, metadata) <= 2
        for candidate in candidates
    )


def test_time_scaling_cap_filters_outer_product_first_path(chain) -> None:
    node, metadata, *_ = chain

    candidates = parenthesization_candidates(
        node,
        metadata,
        scaling_limits=ScalingLimits(time={"n": 3}),
        exhaustive_limit=3,
    )

    assert len(candidates) == 2
    assert all(candidate.path[0] in {(0, 1), (1, 2)} for candidate in candidates)


def test_tuple_space_scaling_caps_count_axes_across_spaces(chain) -> None:
    node, _, *_ = chain
    metadata = Metadata(
        indices={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"]},
        sizes={"I": 2, "J": 2, "K": 2, "L": 2},
    )

    candidates = parenthesization_candidates(
        node,
        metadata,
        scaling_limits=ScalingLimits(time={("I", "J", "K"): 3}, memory={("I", "J", "K"): 2}),
        exhaustive_limit=3,
    )

    assert len(candidates) == 2
    assert all(candidate.path[0] in {(0, 1), (1, 2)} for candidate in candidates)


def test_time_and_memory_caps_can_reject_every_path(chain) -> None:
    node, metadata, *_ = chain

    candidates = parenthesization_candidates(
        node,
        metadata,
        scaling_limits=ScalingLimits(time={"n": 2}),
        exhaustive_limit=3,
    )

    assert candidates == []

    candidates = parenthesization_candidates(
        node,
        metadata,
        scaling_limits=ScalingLimits(memory={"n": 1}),
        exhaustive_limit=3,
    )

    assert candidates == []


def test_dp_respects_time_and_memory_caps(chain) -> None:
    node, metadata, *_ = chain
    limits = ScalingLimits(time={"n": 3}, memory={"n": 2})

    candidates = parenthesization_candidates(node, metadata, scaling_limits=limits, exhaustive_limit=2)

    assert len(candidates) == 1
    assert candidates[0].path[0] in {(0, 1), (1, 2)}

    too_small = parenthesization_candidates(
        node,
        metadata,
        scaling_limits=ScalingLimits(time={"n": 2}),
        exhaustive_limit=2,
    )
    assert too_small == []


def test_scaling_limits_validate_keys_and_values() -> None:
    with pytest.raises(ValueError, match="keys"):
        ScalingLimits(time={(): 2})
    with pytest.raises(ValueError, match="nonnegative integers"):
        ScalingLimits(memory={"n": -1})


def test_scaling_exponent_uses_metadata_spaces(chain) -> None:
    node, metadata, *_ = chain

    assert scaling_exponent(node, "n", ScalingMetric.TIME, metadata) == 3
    assert scaling_exponent(node, ("n",), ScalingMetric.MEMORY, metadata) == 2
    assert scaling_exponent(node, "missing", ScalingMetric.MEMORY, metadata) == 0
    assert not hasattr(node, "scaling_exponent")


def test_node_scaling_exponent_handles_tensor_trace() -> None:
    trace = Tensor("T", ["i", "i", "j"])
    metadata = Metadata(indices={"n": ["i", "j"]}, sizes={"n": 2})

    assert scaling_exponent(trace, "n", ScalingMetric.TIME, metadata) == 2
    assert scaling_exponent(trace, "n", ScalingMetric.MEMORY, metadata) == 2


def test_scaling_exponent_accepts_expression(chain) -> None:
    from albert import Expression

    node, metadata, a, *_ = chain
    expression = Expression(Tensor("R", ["i", "l"]), node)

    assert scaling_exponent(expression, "n", ScalingMetric.TIME, metadata) == scaling_exponent(
        expression.rhs, "n", ScalingMetric.TIME, metadata
    )


def test_scaling_memory_limit_includes_input_tensor_storage() -> None:
    a = Tensor("A", ["i", "j", "k"])
    b = Tensor("B", ["k", "l"])
    node = Mul([a, b])
    metadata = Metadata(indices={"n": ["i", "j", "k", "l"]}, sizes={"n": 4})

    assert scaling_exponent(node, "n", ScalingMetric.MEMORY, metadata) == 3
    assert (
        parenthesization_candidates(
            node,
            metadata,
            scaling_limits=ScalingLimits(memory={"n": 2}),
            exhaustive_limit=2,
        )
        == []
    )


def test_node_scaling_exponent_validates_kind(chain) -> None:
    node, metadata, *_ = chain

    with pytest.raises(ValueError, match="which"):
        scaling_exponent(node, "n", "operations", metadata)
