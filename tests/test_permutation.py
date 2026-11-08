import pytest

from albert import Add, Mul, Permutation, PhaseFactor, Scalar, Tensor


def test_permutation_stores_the_order_and_defaults_to_positive_phase() -> None:
    permutation = Permutation([2, 0, 1])

    assert permutation.permutation == [2, 0, 1]
    assert permutation.phase is PhaseFactor.POSITIVE


def test_permutation_accepts_negative_phase() -> None:
    permutation = Permutation([1, 0], PhaseFactor.NEGATIVE)

    assert permutation.permutation == [1, 0]
    assert permutation.phase is PhaseFactor.NEGATIVE


def test_node_permute_indices_accepts_a_permutation_or_sequence() -> None:
    expression = Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j", "k"])])
    expected = Mul([Tensor("A", ["k", "j"]), Tensor("B", ["j", "i"])])

    by_class = expression.permute_indices(Permutation([1, 0], PhaseFactor.NEGATIVE))
    by_sequence = expression.permute_indices([1, 0])

    assert by_class == expected
    assert by_sequence == expected
    assert by_class.children[0].indices[1] == "j"


def test_node_permute_indices_checks_external_index_arity() -> None:
    tensor = Tensor("A", ["i", "j"])

    with pytest.raises(ValueError, match="number of external indices"):
        tensor.permute_indices([0])


def test_node_permute_indices_applies_a_three_index_cycle() -> None:
    tensor = Tensor("T", ["i", "j", "k"])

    result = tensor.permute_indices([1, 2, 0])

    assert result == Tensor("T", ["j", "k", "i"])


def test_node_permute_indices_updates_every_add_operand() -> None:
    expression = Add([Tensor("A", ["i", "j"]), Tensor("B", ["j", "i"])])

    result = expression.permute_indices(Permutation([1, 0]))

    assert result == Add([Tensor("A", ["j", "i"]), Tensor("B", ["i", "j"])])


@pytest.mark.parametrize("node", [Scalar(1.0), Add([]), Mul([])])
def test_node_permute_indices_handles_nodes_without_external_indices(node) -> None:
    assert node.permute_indices([]) is node


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
