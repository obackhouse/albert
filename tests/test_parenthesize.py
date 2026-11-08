import pytest

from albert import (
    CostModel,
    Metadata,
    Mul,
    ScalingLimits,
    Tensor,
    apply_parenthesization,
    parenthesization_candidates,
)


@pytest.fixture
def chain():
    a = Tensor("A", ["i", "j"])
    b = Tensor("B", ["j", "k"])
    c = Tensor("C", ["k", "l"])
    metadata = Metadata(spaces={"n": ["i", "j", "k", "l"]}, sizes={"n": 10})
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


def test_exhaustive_finds_the_lowest_cost_parenthesization() -> None:
    a = Tensor("A", ["i", "j"])
    b = Tensor("B", ["j", "k"])
    c = Tensor("C", ["k", "l"])
    node = Mul([a, b, c])
    metadata = Metadata(
        spaces={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"]},
        sizes={"I": 2, "J": 2, "K": 10_000, "L": 1_000},
    )
    model = CostModel(multiply=1, add=0, read=0, write=0)

    candidates = parenthesization_candidates(node, metadata, max_candidates=10, exhaustive_limit=3, cost_model=model)

    assert len(candidates) == 3
    assert candidates[0].path == [(1, 2), (0, 1)]
    assert candidates[0].cost.total == min(candidate.cost.total for candidate in candidates)
    assert candidates[0].cost.total < candidates[1].cost.total


def test_apply_parenthesization_builds_binary_product_tree(chain) -> None:
    node, metadata, a, b, c = chain
    candidate = parenthesization_candidates(node, metadata, exhaustive_limit=3)[0]

    result = apply_parenthesization(node, candidate)

    assert isinstance(result, Mul)
    assert len(result.children) == 2
    products = result.search(lambda node: isinstance(node, Mul))
    assert len(products) == 2
    assert all(len(product.children) == 2 for product in products)
    assert result.external_indices == ["i", "l"]
    assert {a.symbol, b.symbol, c.symbol} == {
        tensor.symbol for tensor in result.search(lambda n: isinstance(n, Tensor))
    }


def test_dp_returns_one_path_even_when_max_candidates_is_zero(chain) -> None:
    node, metadata, *_ = chain

    candidates = parenthesization_candidates(node, metadata, max_candidates=0, exhaustive_limit=2)

    assert len(candidates) == 1
    assert len(candidates[0].path) == 2


def test_scaling_caps_filter_exhaustive_paths(chain) -> None:
    node, metadata, *_ = chain
    limits = ScalingLimits(time={"n": 3}, memory={"n": 2})

    candidates = parenthesization_candidates(node, metadata, scaling_limits=limits, exhaustive_limit=3)

    assert len(candidates) == 2
    assert all(candidate.path[0] in {(0, 1), (1, 2)} for candidate in candidates)
    assert all(
        apply_parenthesization(node, candidate).scaling_exponent("n", "time", metadata) <= 3 for candidate in candidates
    )
    assert all(
        apply_parenthesization(node, candidate).scaling_exponent("n", "memory", metadata) <= 2
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
        spaces={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"]},
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


def test_node_scaling_exponent_uses_metadata_spaces(chain) -> None:
    node, metadata, *_ = chain

    assert node.scaling_exponent("n", "time", metadata) == 3
    assert node.scaling_exponent(("n",), "memory", metadata) == 2
    assert node.scaling_exponent("missing", "memory", metadata) == 0


def test_node_scaling_exponent_handles_tensor_trace() -> None:
    trace = Tensor("T", ["i", "i", "j"])
    metadata = Metadata(spaces={"n": ["i", "j"]})

    assert trace.scaling_exponent("n", "time", metadata) == 2
    assert trace.scaling_exponent("n", "memory", metadata) == 2


def test_scaling_memory_limit_includes_input_tensor_storage() -> None:
    a = Tensor("A", ["i", "j", "k"])
    b = Tensor("B", ["k", "l"])
    node = Mul([a, b])
    metadata = Metadata(spaces={"n": ["i", "j", "k", "l"]}, sizes={"n": 4})

    assert node.scaling_exponent("n", "memory", metadata) == 3
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
        node.scaling_exponent("n", "operations", metadata)
