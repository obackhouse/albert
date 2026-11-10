from albert import Metadata, Permutation


def test_metadata_add_space_stores_indices_and_default_size() -> None:
    metadata = Metadata()

    metadata.add_space("occupied", ("i", "j"))

    assert metadata.spaces == {"occupied": ["i", "j"]}
    assert metadata.sizes == {"occupied": 8}


def test_metadata_add_space_stores_a_custom_size() -> None:
    metadata = Metadata()

    metadata.add_space("virtual", ["a", "b", "c"], size=24)

    assert metadata.spaces["virtual"] == ["a", "b", "c"]
    assert metadata.sizes["virtual"] == 24


def test_metadata_add_tensor_stores_default_cost_factor() -> None:
    metadata = Metadata()

    metadata.add_tensor("T")

    assert metadata.cost_factor == {"T": 1}
    assert metadata.symmetries == {}


def test_metadata_add_tensor_stores_cost_factor_and_symmetry() -> None:
    metadata = Metadata()
    symmetry = Permutation([1, 0])

    metadata.add_tensor("A", symmetry=(symmetry,), cost_factor=3)

    assert metadata.cost_factor == {"A": 3}
    assert metadata.symmetries == {"A": [symmetry]}
