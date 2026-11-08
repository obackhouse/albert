import pytest

from albert import Add, Expression, Scalar, Tensor


def test_expression_wraps_tensor_lhs_and_node_rhs() -> None:
    lhs = Tensor("R", ["i", "j"])
    rhs = Add([Tensor("A", ["i", "j"]), Tensor("B", ["i", "j"])])

    expression = Expression(lhs, rhs)

    assert expression.lhs is lhs
    assert expression.rhs is rhs


def test_expression_requires_a_tensor_lhs() -> None:
    with pytest.raises(TypeError):
        Expression(Scalar(1), Tensor("A", []))  # type: ignore[arg-type]
