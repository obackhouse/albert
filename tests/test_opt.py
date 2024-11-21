from albert.opt.tools import substitute_expressions
from albert.tensor import Tensor


def test_substitute_expressions():
    z_output = Tensor.from_string("z(i)")
    z_expr = Tensor.from_string("x(i,j) * y(j)")
    x_output = Tensor.from_string("x(i,j)")
    x_expr = Tensor.from_string("a(i,k,l) * b(k,l,j)")
    output_expr = [(z_output, z_expr), (x_output, x_expr)]
    output_expr_sub = substitute_expressions(output_expr)
    assert len(output_expr_sub) == 1
    assert output_expr_sub[0][0] == z_output
    assert output_expr_sub[0][1] == Tensor.from_string("a(i,j,k) * b(j,k,l) * y(l)").expand()

    x_output = Tensor.from_string("x(i,j)")
    x_expr = Tensor.from_string("a(i,k,l) * b(k,l,j)")
    y_output = Tensor.from_string("y(i,j)")
    y_expr = Tensor.from_string("a(i,k,l) * c(k,l,j)")
    u_output = Tensor.from_string("u(i,j)")
    u_expr = Tensor.from_string("x(i,j) + z(i,j)")
    v_output = Tensor.from_string("v(i,j)")
    v_expr = Tensor.from_string("y(i,j) + z(i,j)")
    output_expr = [
        (x_output, x_expr),
        (y_output, y_expr),
        (u_output, u_expr),
        (v_output, v_expr),
    ]
    output_expr_sub = substitute_expressions(output_expr)
    assert len(output_expr_sub) == 2
    assert output_expr_sub[0][0] == u_output
    assert output_expr_sub[0][1] == Tensor.from_string("(a(i,k,l) * b(k,l,j)) + (z(i,j))").expand()
    assert output_expr_sub[1][0] == v_output
    assert output_expr_sub[1][1] == Tensor.from_string("(a(i,k,l) * c(k,l,j)) + (z(i,j))").expand()

    x_output = Tensor.from_string("x(i,j)")
    x_expr = Tensor.from_string("a(i,k,l) * b(k,l,j)")
    y_output = Tensor.from_string("y(i,j)")
    y_expr = Tensor.from_string("a(i,k,l) * c(k,l,j)")
    u_output = Tensor.from_string("u(i,j)")
    u_expr = Tensor.from_string("x(i,j) + y(i,j) + z(i,j)")
    v_output = Tensor.from_string("v(i,j)")
    v_expr = Tensor.from_string("x(i,j) + y(i,j) + z(i,j)")
    output_expr = [
        (x_output, x_expr),
        (y_output, y_expr),
        (u_output, u_expr),
        (v_output, v_expr),
    ]
    output_expr_sub = substitute_expressions(output_expr)
    assert len(output_expr_sub) == 2
    assert output_expr_sub[0][0] == u_output
    assert (
        output_expr_sub[0][1]
        == Tensor.from_string("(a(i,k,l) * b(k,l,j)) + (a(i,k,l) * c(k,l,j)) + (z(i,j))").expand()
    )
    assert output_expr_sub[1][0] == v_output
    assert (
        output_expr_sub[1][1]
        == Tensor.from_string("(a(i,k,l) * b(k,l,j)) + (a(i,k,l) * c(k,l,j)) + (z(i,j))").expand()
    )
