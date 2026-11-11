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


def test_metadata_add_tensor_stores_symmetry() -> None:
    metadata = Metadata()
    symmetry = Permutation([1, 0])

    metadata.add_space("occupied", ("i", "j"))
    metadata.add_tensor("A", symmetry=(symmetry,))

    assert metadata.symmetries == {"A": [symmetry]}


def test_metadata_add_tensor_uses_tensor_symbol_keys() -> None:
    metadata = Metadata()
    symmetry = Permutation([1, 0])

    metadata.add_space("occupied", ("i", "j"))
    metadata.add_tensor("A", symmetry=(symmetry,))

    assert metadata.symmetries == {"A": [symmetry]}


def test_metadata_rejects_invalid_tensor_metadata_keys() -> None:
    with pytest.raises(ValueError, match="tensor metadata keys"):
        Metadata(symmetries={"": []})
