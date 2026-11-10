from albert import Tensor
from albert.factory import TensorFactory


def test_tensor_factory_builds_a_tensor_from_a_single_index() -> None:
    factory = TensorFactory("A")

    tensor = factory["i"]

    assert tensor == Tensor("A", ["i"])


def test_tensor_factory_builds_a_tensor_from_multiple_indices() -> None:
    factory = TensorFactory("A")

    tensor = factory["i", "j", "k"]

    assert tensor == Tensor("A", ["i", "j", "k"])


def test_tensor_factory_preserves_a_string_index_as_one_index() -> None:
    factory = TensorFactory("A")

    assert factory["ij"] == Tensor("A", ["ij"])


def test_tensor_factory_builds_a_rank_zero_tensor() -> None:
    factory = TensorFactory("E")

    assert factory[()] == Tensor("E", [])
