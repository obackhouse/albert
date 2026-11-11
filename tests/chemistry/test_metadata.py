import pytest

from albert.chemistry.metadata import Metadata


def test_chemistry_metadata_has_default_spaces_and_sizes() -> None:
    metadata = Metadata()

    assert metadata.indices["o"][:4] == ["i", "j", "k", "l"]
    assert metadata.indices["v"][:4] == ["a", "b", "c", "d"]
    assert metadata.indices["x"][:4] == ["P", "Q", "R", "S"]
    assert metadata.sizes == {"o": 64, "v": 512, "x": 1728}


def test_chemistry_metadata_includes_default_tensor_symmetries() -> None:
    metadata = Metadata()

    assert set(metadata.symmetries) == {
        "h",
        "f",
        "v",
        "v_a",
        "t2",
        "t3",
        "t2_a",
        "t3_a",
        "l2",
        "l3",
        "l2_a",
        "l3_a",
        "d1",
        "d2_a",
    }


def test_with_spin_adds_alpha_beta_and_spin_spaces_with_sizes() -> None:
    metadata = Metadata().with_spin()

    assert metadata.indices["o_α"][:4] == [f"{index}_α" for index in ["i", "j", "k", "l"]]
    assert metadata.indices["o_β"][:4] == [f"{index}_β" for index in ["i", "j", "k", "l"]]
    assert metadata.indices["o_γ"][:4] == [f"{index}_γ" for index in ["i", "j", "k", "l"]]
    assert metadata.sizes["o_α"] == metadata.sizes["o_β"] == 64
    assert metadata.sizes["o_γ"] == 2 * 64
    assert metadata.sizes["v_γ"] == 2 * 512
    assert metadata.indices["x_α"][0] == "P_α"


def test_with_spin_rejects_already_spin_labeled_metadata() -> None:
    metadata = Metadata(indices={"o_α": ["i_α"]}, sizes={"o_α": 8})

    with pytest.raises(ValueError, match="already has spin-aware indices"):
        metadata.with_spin()
