import pytest

from albert import Tensor
from albert.chemistry.factory import TensorFactory


def test_factory_provides_tensor_rank_description_and_symmetry() -> None:
    factory = TensorFactory("v")

    assert factory.rank == 4
    assert factory.description == "Two-electron integrals"
    assert factory.symmetry()


def test_factory_builds_tensors_and_checks_rank() -> None:
    factory = TensorFactory("t1")

    assert factory["i", "a"] == Tensor("t1", ["i", "a"])
    with pytest.raises(ValueError, match="expects 2 indices"):
        factory["i"]


def test_factory_rejects_unknown_symbols() -> None:
    with pytest.raises(ValueError, match="unsupported tensor symbol"):
        TensorFactory("unknown")
