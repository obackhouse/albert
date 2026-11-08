import math

import pytest

from albert import Add, CostModel, Metadata, Mul, Scalar, Tensor, estimate_cost


def test_cost_estimate_for_contraction_counts_arithmetic_and_traffic() -> None:
    metadata = Metadata(
        spaces={"occupied": ["i", "k"], "virtual": ["j"]},
        sizes={"occupied": 2, "virtual": 3},
    )
    expression = Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j", "k"])])

    estimate = estimate_cost(expression, metadata)

    assert estimate.multiplications == 12
    assert estimate.additions == 8
    assert estimate.reads == 12
    assert estimate.writes == 4
    assert estimate.total == pytest.approx(21.6)


def test_cost_model_weights_arithmetic_and_memory_traffic_independently() -> None:
    metadata = Metadata(
        spaces={"occupied": ["i", "k"], "virtual": ["j"]},
        sizes={"occupied": 2, "virtual": 3},
    )
    expression = Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j", "k"])])
    model = CostModel(multiply=2.0, add=3.0, read=0.5, write=0.25)

    estimate = estimate_cost(expression, metadata, model)

    assert estimate.multiplications == 12
    assert estimate.additions == 8
    assert estimate.reads == 12
    assert estimate.writes == 4
    assert estimate.total == pytest.approx(55.0)


def test_cost_estimate_for_elementwise_addition() -> None:
    metadata = Metadata(spaces={"occupied": ["i"]}, sizes={"occupied": 7})
    expression = Add([Tensor("A", ["i"]), Tensor("B", ["i"])])

    estimate = estimate_cost(expression, metadata)

    assert estimate.multiplications == 0
    assert estimate.additions == 7
    assert estimate.reads == 14
    assert estimate.writes == 7
    assert estimate.total == pytest.approx(9.1)


def test_cost_estimate_includes_intermediate_traffic_for_a_contraction_chain() -> None:
    metadata = Metadata(
        spaces={"occupied": ["i", "k"], "virtual": ["j", "l"]},
        sizes={"occupied": 2, "virtual": 3},
    )
    a = Tensor("A", ["i", "j"])
    b = Tensor("B", ["j", "k"])
    c = Tensor("C", ["k", "l"])

    estimate = estimate_cost(Mul([Mul([a, b]), c]), metadata)

    assert estimate.multiplications == 24
    assert estimate.additions == 14
    assert estimate.reads == 22
    assert estimate.writes == 10
    assert estimate.total == pytest.approx(41.2)


def test_cost_estimate_counts_scalar_broadcast_multiplication() -> None:
    metadata = Metadata(spaces={"occupied": ["i"], "virtual": ["j"]}, sizes={"occupied": 2, "virtual": 3})

    estimate = estimate_cost(Mul([Scalar(2), Tensor("A", ["i", "j"])]), metadata)

    assert estimate.multiplications == 6
    assert estimate.additions == 0
    assert estimate.reads == 7
    assert estimate.writes == 6
    assert estimate.total == pytest.approx(7.3)


def test_cost_estimate_counts_a_tensor_trace() -> None:
    metadata = Metadata(spaces={"occupied": ["i"]}, sizes={"occupied": 4})

    estimate = estimate_cost(Tensor("A", ["i", "i"]), metadata)

    assert estimate.multiplications == 0
    assert estimate.additions == 3
    assert estimate.reads == 4
    assert estimate.writes == 1
    assert estimate.total == pytest.approx(3.5)


def test_scalar_cost_does_not_depend_on_scalar_value_or_metadata() -> None:
    estimate = estimate_cost(Scalar(-7), Metadata())

    assert (
        estimate.multiplications,
        estimate.additions,
        estimate.reads,
        estimate.writes,
        estimate.total,
    ) == (0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("spaces", "sizes", "indices", "message"),
    [
        ({"occupied": ["i"]}, {"occupied": 2}, ["unknown"], "does not belong"),
        ({"occupied": ["i"]}, {}, ["i"], "no declared size"),
        ({"occupied": ["i"], "virtual": ["i"]}, {"occupied": 2, "virtual": 3}, ["i"], "multiple spaces"),
        ({"occupied": ["i"]}, {"occupied": 0}, ["i"], "positive"),
    ],
)
def test_cost_estimate_rejects_invalid_metadata(spaces, sizes, indices, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_cost(Tensor("A", indices), Metadata(spaces=spaces, sizes=sizes))


@pytest.mark.parametrize("weight", [-1.0, math.inf, math.nan])
def test_cost_model_rejects_negative_or_nonfinite_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        CostModel(multiply=weight)
