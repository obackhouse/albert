from dataclasses import FrozenInstanceError

import pytest

from albert import Add, Metadata, Mul, NodeKind, Scalar, Tensor


def test_metadata_stores_space_to_index_names() -> None:
    metadata = Metadata(spaces={"occupied": ["i", "j"], "virtual": ["a", "b"]})

    assert metadata.spaces["occupied"] == ["i", "j"]
    assert metadata.spaces["virtual"] == ["a", "b"]
    with pytest.raises(FrozenInstanceError):
        metadata.spaces = {}  # type: ignore[misc]


def test_tensor_keeps_order_and_duplicate_indices() -> None:
    value = Tensor("T", ["i", "k", "i"])

    assert isinstance(value, Tensor)
    assert value.symbol == "T"
    assert value.rank == 3
    assert value.indices == ["i", "k", "i"]


def test_add_mul_store_their_children() -> None:
    first = Scalar(2.0)
    second = Scalar(3.0)
    expression = Add([first, second])
    product = Mul([first, second])

    assert isinstance(expression, Add)
    assert expression.kind == NodeKind.ADD
    assert expression.arity == 2
    assert expression.children == [first, second]
    assert isinstance(product, Mul)
    assert product.kind == NodeKind.MUL
    assert product.arity == 2
    assert product.children == [first, second]


def test_add_requires_the_same_free_indices() -> None:
    ordered = Add([Tensor("A", ["i", "j"]), Tensor("B", ["j", "i"])])
    contracted = Mul([Tensor("C", ["i", "j"]), Tensor("D", ["j", "k"])])
    product_sum = Add([contracted, Tensor("E", ["k", "i"])])

    assert ordered.arity == 2
    assert product_sum.arity == 2
    with pytest.raises(ValueError, match="same free indices"):
        Add([Tensor("A", ["i"]), Tensor("B", ["j"])])
    with pytest.raises(ValueError, match="same free indices"):
        Add([Scalar(1.0), Tensor("A", ["i"])])


def test_metadata_is_separate_from_nodes() -> None:
    metadata = Metadata(spaces={"occupied": ["i", "j"]})
    value = Tensor("A", [])

    assert isinstance(value, Tensor)
    assert not hasattr(value, "metadata")
    assert metadata.spaces["occupied"] == ["i", "j"]
