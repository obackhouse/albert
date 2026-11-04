import numpy as np
import pytest

from albert.expression import Expression
from albert.opt.tools import sort_expressions, substitute_expressions
from albert.opt import optimise
from albert.tensor import Tensor
from albert.index import from_list


def test_substitute_expressions():
    z_output = Tensor.from_string("z(i)")
    z_expr = Tensor.from_string("x(i,j) * y(j)")
    x_output = Tensor.from_string("x(i,j)")
    x_expr = Tensor.from_string("a(i,k,l) * b(k,l,j)")
    output_expr = [Expression(z_output, z_expr), Expression(x_output, x_expr)]
    output_expr_sub = substitute_expressions(output_expr)
    assert len(output_expr_sub) == 1
    assert output_expr_sub[0].lhs == z_output
    assert output_expr_sub[0].rhs == Tensor.from_string("a(i,j,k) * b(j,k,l) * y(l)").expand()

    x_output = Tensor.from_string("x(i,j)")
    x_expr = Tensor.from_string("a(i,k,l) * b(k,l,j)")
    y_output = Tensor.from_string("y(i,j)")
    y_expr = Tensor.from_string("a(i,k,l) * c(k,l,j)")
    u_output = Tensor.from_string("u(i,j)")
    u_expr = Tensor.from_string("x(i,j) + z(i,j)")
    v_output = Tensor.from_string("v(i,j)")
    v_expr = Tensor.from_string("y(i,j) + z(i,j)")
    output_expr = [
        Expression(x_output, x_expr),
        Expression(y_output, y_expr),
        Expression(u_output, u_expr),
        Expression(v_output, v_expr),
    ]
    output_expr_sub = substitute_expressions(output_expr)
    assert len(output_expr_sub) == 2
    assert output_expr_sub[0].lhs == u_output
    assert output_expr_sub[0].rhs == Tensor.from_string("(a(i,k,l) * b(k,l,j)) + (z(i,j))").expand()
    assert output_expr_sub[1].lhs == v_output
    assert output_expr_sub[1].rhs == Tensor.from_string("(a(i,k,l) * c(k,l,j)) + (z(i,j))").expand()

    x_output = Tensor.from_string("x(i,j)")
    x_expr = Tensor.from_string("a(i,k,l) * b(k,l,j)")
    y_output = Tensor.from_string("y(i,j)")
    y_expr = Tensor.from_string("a(i,k,l) * c(k,l,j)")
    u_output = Tensor.from_string("u(i,j)")
    u_expr = Tensor.from_string("x(i,j) + y(i,j) + z(i,j)")
    v_output = Tensor.from_string("v(i,j)")
    v_expr = Tensor.from_string("x(i,j) + y(i,j) + z(i,j)")
    output_expr = [
        Expression(x_output, x_expr),
        Expression(y_output, y_expr),
        Expression(u_output, u_expr),
        Expression(v_output, v_expr),
    ]
    output_expr_sub = substitute_expressions(output_expr)
    assert len(output_expr_sub) == 2
    assert output_expr_sub[0].lhs == u_output
    assert (
        output_expr_sub[0].rhs
        == Tensor.from_string("(a(i,k,l) * b(k,l,j)) + (a(i,k,l) * c(k,l,j)) + (z(i,j))").expand()
    )
    assert output_expr_sub[1].lhs == v_output
    assert (
        output_expr_sub[1].rhs
        == Tensor.from_string("(a(i,k,l) * b(k,l,j)) + (a(i,k,l) * c(k,l,j)) + (z(i,j))").expand()
    )


@pytest.mark.parametrize("method", ["auto", "gristmill", "albert", "legacy"])
def test_optimise(method: str):
    i, j, k, l = from_list(["i", "j", "k", "l"], spaces="o")
    lhs = Tensor(i, j, name="x")
    rhs = (
        Tensor(i, k, l, name="a") * Tensor(k, l, j, name="b")
        + Tensor(i, k, l, name="a") * Tensor(k, l, j, name="c")
    )
    expr = Expression(lhs, rhs)
    optimised_exprs = optimise([expr], method=method)
    expr_recovered = substitute_expressions(optimised_exprs)[0]
    assert len(optimised_exprs) == 2
    assert expr_recovered.lhs == expr.lhs


def _evaluate_schedule(expressions, arrays):
    """Evaluate an ordered tensor schedule with NumPy."""
    values = dict(arrays)
    assigned = set()
    for expression in sort_expressions(expressions):
        value = expression.rhs.evaluate(values, np.einsum)
        permutation = tuple(
            expression.rhs.external_indices.index(index) for index in expression.lhs.indices
        )
        value = value.transpose(permutation)
        if expression.lhs.name in assigned:
            values[expression.lhs.name] += value
        else:
            values[expression.lhs.name] = value
            assigned.add(expression.lhs.name)
    return values


def test_optimise_albert_preserves_intermediate_axis_order():
    i, j, k, l = from_list(["i", "j", "k", "l"])
    lhs = Tensor(i, j, name="r")
    rhs = Tensor(i, k, l, name="a") * (
        Tensor(k, l, j, name="b") + Tensor(k, l, j, name="c")
    )
    expr = Expression(lhs, rhs)

    rng = np.random.default_rng(123)
    arrays = {name: rng.normal(size=(3, 3, 3)) for name in ("a", "b", "c")}
    expected = np.einsum("ikl,klj->ij", arrays["a"], arrays["b"]) + np.einsum(
        "ikl,klj->ij", arrays["a"], arrays["c"]
    )

    optimised = optimise([expr], method="albert", sizes={None: 3})
    actual = _evaluate_schedule(optimised, arrays)["r"]

    assert np.allclose(actual, expected)


def test_optimise_albert_does_not_merge_partial_bicliques_across_outputs():
    i, j, k, l = from_list(["i", "j", "k", "l"])
    a = Tensor(i, k, l, name="a")
    b = Tensor(k, l, j, name="b")
    c = Tensor(k, l, j, name="c")
    d = Tensor(k, l, j, name="d")
    expressions = [
        Expression(Tensor(i, j, name="r1"), a * b + a * c),
        Expression(Tensor(i, j, name="r2"), a * b + a * d),
    ]

    rng = np.random.default_rng(456)
    arrays = {name: rng.normal(size=(3, 3, 3)) for name in ("a", "b", "c", "d")}
    expected = {
        "r1": np.einsum("ikl,klj->ij", arrays["a"], arrays["b"])
        + np.einsum("ikl,klj->ij", arrays["a"], arrays["c"]),
        "r2": np.einsum("ikl,klj->ij", arrays["a"], arrays["b"])
        + np.einsum("ikl,klj->ij", arrays["a"], arrays["d"]),
    }

    optimised = optimise(expressions, method="albert", sizes={None: 3})
    actual = _evaluate_schedule(optimised, arrays)

    assert np.allclose(actual["r1"], expected["r1"])
    assert np.allclose(actual["r2"], expected["r2"])


def test_optimise_albert_preserves_matrix_chain_contractions():
    i, j, k, l = from_list(["i", "j", "k", "l"])
    expr = Expression(
        Tensor(i, j, name="r"),
        Tensor(i, k, name="a") * Tensor(k, l, name="b") * Tensor(l, j, name="c"),
    )

    rng = np.random.default_rng(789)
    arrays = {name: rng.normal(size=(3, 3)) for name in ("a", "b", "c")}
    expected = np.einsum("ik,kl,lj->ij", arrays["a"], arrays["b"], arrays["c"])

    optimised = optimise([expr], method="albert", sizes={None: 3})
    actual = _evaluate_schedule(optimised, arrays)["r"]

    assert np.allclose(actual, expected)
