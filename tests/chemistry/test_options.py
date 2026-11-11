import pytest

from albert.chemistry.options import T2_ANTISYMMETRY, T2_SYMMETRY, OrbitalType, ReferenceType


@pytest.mark.parametrize(
    ("reference", "prefix"),
    [
        (ReferenceType.RHF, "r"),
        (ReferenceType.UHF, "u"),
        (ReferenceType.GHF, "g"),
    ],
)
def test_reference_types_expose_their_prefix(reference: ReferenceType, prefix: str) -> None:
    assert reference.prefix == prefix


@pytest.mark.parametrize(
    ("orbital", "subscript", "size_factor"),
    [
        (OrbitalType.ALPHA, "_α", 1),
        (OrbitalType.BETA, "_β", 1),
        (OrbitalType.SPIN, "_γ", 2),
        (OrbitalType.SPATIAL, "", 1),
    ],
)
def test_orbital_types_define_index_suffix_and_size_factor(
    orbital: OrbitalType,
    subscript: str,
    size_factor: int,
) -> None:
    assert orbital.subscript == subscript
    assert orbital.size_factor == size_factor


def test_t2_symmetries_preserve_the_occupied_and_virtual_slots() -> None:
    assert [symmetry.permutation for symmetry in T2_SYMMETRY] == [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
    ]
    assert [symmetry.permutation for symmetry in T2_ANTISYMMETRY] == [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 3, 2],
    ]
