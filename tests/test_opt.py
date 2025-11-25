import pytest

from albert.expression import Expression
from albert.opt.tools import substitute_expressions
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
