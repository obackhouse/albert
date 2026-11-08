import pytest

from albert import Permutation, PhaseFactor


def test_permutation_stores_the_order_and_defaults_to_positive_phase() -> None:
    permutation = Permutation([2, 0, 1])

    assert permutation.permutation == [2, 0, 1]
    assert permutation.phase is PhaseFactor.POSITIVE


def test_permutation_accepts_negative_phase() -> None:
    permutation = Permutation([1, 0], PhaseFactor.NEGATIVE)

    assert permutation.permutation == [1, 0]
    assert permutation.phase is PhaseFactor.NEGATIVE


def test_empty_permutation_is_valid() -> None:
    permutation = Permutation([])

    assert permutation.permutation == []
    assert permutation.phase is PhaseFactor.POSITIVE


@pytest.mark.parametrize("values", [[0, 0], [0, 2], [-1, 0]])
def test_permutation_rejects_invalid_orders(values: list[int]) -> None:
    with pytest.raises(ValueError, match="permutation must contain"):
        Permutation(values)


@pytest.mark.parametrize("phase", [0, 2])
def test_permutation_rejects_invalid_phase(phase: int) -> None:
    with pytest.raises(ValueError, match="phase must be"):
        Permutation([0], phase)  # type: ignore[arg-type]
