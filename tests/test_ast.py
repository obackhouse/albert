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


def test_leaf_nodes_expose_empty_children() -> None:
    assert Scalar(2.0).children == []
    assert Tensor("T", ["i", "j"]).children == []


def test_node_exposes_external_internal_and_tensor_indices() -> None:
    tensor = Tensor("A", ["i", "i", "j"])
    scalar = Scalar(2.0)
    product = Mul(
        [
            Tensor("A", ["i", "k"]),
            Tensor("B", ["k", "j"]),
            Tensor("C", ["j", "l"]),
        ],
    )
    addition = Add([Tensor("X", ["a", "b"]), Tensor("Y", ["b", "a"])])

    assert tensor.indices == ["i", "i", "j"]
    assert tensor.external_indices == ["j"]
    assert tensor.internal_indices == ["i", "i"]
    assert scalar.external_indices == []
    assert scalar.internal_indices == []
    assert addition.external_indices == ["a", "b"]
    assert addition.internal_indices == []
    assert product.external_indices == ["i", "l"]
    assert product.internal_indices == ["k", "j"]


def test_empty_and_unary_nodes_expose_index_properties() -> None:
    empty_add = Add([])
    empty_mul = Mul([])
    tensor = Tensor("A", ["i"])
    unary_add = Add([tensor])
    unary_mul = Mul([tensor])

    for node in (empty_add, empty_mul):
        assert node.external_indices == []
        assert node.internal_indices == []
    for node in (unary_add, unary_mul):
        assert node.external_indices == ["i"]
        assert node.internal_indices == []


def test_map_indices_swaps_tensor_indices_simultaneously() -> None:
    original = Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k", "j"])])

    result = original.map_indices({"i": "j", "j": "i"})

    assert isinstance(result, Mul)
    assert result.children[0].indices == ["j", "k"]
    assert result.children[1].indices == ["k", "i"]
    assert original.children[0].indices == ["i", "k"]


def test_map_indices_rebuilds_nested_structures_and_reuses_unchanged_nodes() -> None:
    scalar = Scalar(2.0)
    a = Tensor("A", ["i", "k"])
    b = Tensor("B", ["k", "i"])
    c = Tensor("C", ["j"])
    d = Tensor("D", ["j"])
    expression = Mul([scalar, Add([a, b]), Mul([c, d])])

    result = expression.map_indices({"i": "k", "k": "i"})
    expected = Mul(
        [
            scalar,
            Add([Tensor("A", ["k", "i"]), Tensor("B", ["i", "k"])]),
            Mul([c, d]),
        ],
    )

    assert result == expected
    assert isinstance(result, Mul)
    assert result.children[0] is scalar
    assert result.children[1] is not expression.children[1]
    assert result.children[2] is expression.children[2]


def test_map_indices_returns_the_original_node_when_no_index_changes() -> None:
    expression = Mul([Tensor("A", ["i"]), Scalar(1.0)])

    assert expression.map_indices({}) is expression


def test_map_indices_rejects_keys_that_are_not_external() -> None:
    expression = Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k", "j"])])

    with pytest.raises(ValueError, match="keys must be external indices"):
        expression.map_indices({"k": "m"})


def test_map_indices_rejects_indices_on_scalar_and_repeated_tensor_axes() -> None:
    scalar = Scalar(2.0)
    trace = Tensor("A", ["i", "i"])

    with pytest.raises(ValueError, match="keys must be external indices"):
        scalar.map_indices({"i": "j"})
    with pytest.raises(ValueError, match="keys must be external indices"):
        trace.map_indices({"i": "j"})


@pytest.mark.parametrize(
    ("node", "mapping", "expected"),
    [
        (Add([]), {}, Add([])),
        (Mul([]), {}, Mul([])),
        (Scalar(1.0), {}, Scalar(1.0)),
        (Tensor("A", ["i"]), {"i": "a"}, Tensor("A", ["a"])),
        (Add([Tensor("A", ["i"])]), {"i": "a"}, Add([Tensor("A", ["a"])])),
        (
            Add([Tensor("A", ["i"]), Tensor("B", ["i"]), Tensor("C", ["i"])]),
            {"i": "a"},
            Add([Tensor("A", ["a"]), Tensor("B", ["a"]), Tensor("C", ["a"])]),
        ),
        (
            Mul([Tensor("A", ["i"]), Tensor("B", ["j"])]),
            {"i": "a", "j": "b"},
            Mul([Tensor("A", ["a"]), Tensor("B", ["b"])]),
        ),
        (
            Mul([Tensor("A", ["i"]), Tensor("B", ["j"]), Tensor("C", ["j"])]),
            {"i": "a"},
            Mul([Tensor("A", ["a"]), Tensor("B", ["j"]), Tensor("C", ["j"])]),
        ),
    ],
)
def test_map_indices_handles_node_types_and_arities(node, mapping, expected) -> None:
    assert node.map_indices(mapping) == expected


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


def test_add_requires_the_same_external_indices() -> None:
    ordered = Add([Tensor("A", ["i", "j"]), Tensor("B", ["j", "i"])])
    internal_product = Mul([Tensor("C", ["i", "j"]), Tensor("D", ["j", "k"])])
    product_sum = Add([internal_product, Tensor("E", ["k", "i"])])

    assert ordered.arity == 2
    assert product_sum.arity == 2
    with pytest.raises(ValueError, match="same external indices"):
        Add([Tensor("A", ["i"]), Tensor("B", ["j"])])
    with pytest.raises(ValueError, match="same external indices"):
        Add([Scalar(1.0), Tensor("A", ["i"])])


def test_metadata_is_separate_from_nodes() -> None:
    metadata = Metadata(spaces={"occupied": ["i", "j"]})
    value = Tensor("A", [])

    assert isinstance(value, Tensor)
    assert not hasattr(value, "metadata")
    assert metadata.spaces["occupied"] == ["i", "j"]
