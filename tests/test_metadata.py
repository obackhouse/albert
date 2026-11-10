import pytest

from albert import Metadata, Permutation


def test_metadata_requires_indices_and_sizes_to_have_the_same_keys() -> None:
    with pytest.raises(ValueError, match="indices and sizes must have the same keys"):
        Metadata(indices={"occupied": ["i"]})


def test_metadata_rejects_duplicate_indices_within_or_across_spaces() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        Metadata(indices={"occupied": ["i", "i"]}, sizes={"occupied": 8})
    with pytest.raises(ValueError, match="must not overlap"):
        Metadata(indices={"occupied": ["i"], "virtual": ["i"]}, sizes={"occupied": 8, "virtual": 8})


def test_metadata_add_space_stores_indices_and_default_size() -> None:
    metadata = Metadata()

    metadata.add_space("occupied", ("i", "j"))

    assert metadata.indices == {"occupied": ["i", "j"]}
    assert metadata.sizes == {"occupied": 8}


def test_metadata_add_space_stores_a_custom_size() -> None:
    metadata = Metadata()

    metadata.add_space("virtual", ["a", "b", "c"], size=24)

    assert metadata.indices["virtual"] == ["a", "b", "c"]
    assert metadata.sizes["virtual"] == 24


def test_metadata_add_tensor_stores_cost_factor_and_symmetry() -> None:
    metadata = Metadata()
    symmetry = Permutation([1, 0])

    metadata.add_space("occupied", ("i", "j"))
    metadata.add_tensor("A", spaces=("occupied", "occupied"), symmetry=(symmetry,), cost_factor=3)

    key = ("A", ("occupied", "occupied"))
    assert metadata.cost_factors == {key: 3}
    assert metadata.symmetries == {key: [symmetry]}


def test_metadata_add_tensor_can_target_an_ordered_space_signature() -> None:
    metadata = Metadata()
    symmetry = Permutation([1, 0])

    metadata.add_space("occupied", ("i", "j"))
    metadata.add_tensor("A", spaces=("occupied", "occupied"), symmetry=(symmetry,), cost_factor=4)

    key = ("A", ("occupied", "occupied"))
    assert metadata.cost_factors == {key: 4}
    assert metadata.symmetries == {key: [symmetry]}


def test_metadata_rejects_invalid_tensor_metadata_keys() -> None:
    with pytest.raises(ValueError, match="tensor metadata keys"):
        Metadata(cost_factors={("", ()): 2})


def test_metadata_requires_spaces_in_tensor_metadata_keys() -> None:
    with pytest.raises(ValueError, match="tensor metadata keys"):
        Metadata(cost_factors={("A", None): 2})  # type: ignore[dict-item]
