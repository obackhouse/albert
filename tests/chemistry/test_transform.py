import pytest

from albert import Tensor
from albert.chemistry import expand_antisymmetrized_eris
from albert.chemistry.factory import TensorFactory
from albert.chemistry.metadata import Metadata


def test_antisymmetrized_eri_expansion_preserves_physicist_index_order() -> None:
    metadata = Metadata()
    i, j = metadata.indices["o"][:2]
    a, b = metadata.indices["v"][:2]
    tensor = TensorFactory("v_a")[i, j, a, b]

    result = tensor.apply(expand_antisymmetrized_eris)

    assert result.format() == f"v[{i}, {j}, {a}, {b}] - v[{i}, {j}, {b}, {a}]"


def test_antisymmetrized_eri_transform_leaves_other_nodes_unchanged() -> None:
    tensor = TensorFactory("v")["i", "j", "a", "b"]

    assert tensor.apply(expand_antisymmetrized_eris) == tensor


def test_antisymmetrized_eri_transform_checks_rank() -> None:
    tensor = Tensor("v_a", ["i", "j"])

    with pytest.raises(ValueError, match="must have rank 4"):
        tensor.apply(expand_antisymmetrized_eris)
