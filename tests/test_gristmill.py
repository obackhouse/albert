import pytest
from albert.index import Index
from albert.tensor import Tensor
from albert.symmetry import Permutation, Symmetry
from albert.opt._gristmill import optimise_gristmill
from albert.opt.tools import substitute_expressions


def test_gristmill():
    i = Index("i")
    j = Index("j")
    k = Index("k")
    l = Index("l")
    symmetry_3c = Symmetry(
        Permutation((0, 1, 2), 1),
        Permutation((0, 2, 1), 1),
        Permutation((1, 0, 2), 1),
        Permutation((1, 2, 0), 1),
        Permutation((2, 0, 1), 1),
        Permutation((2, 1, 0), 1),
    )
    tensor1a = Tensor(i, j, k, name="a", symmetry=symmetry_3c)
    tensor1b = Tensor(i, j, k, name="b", symmetry=symmetry_3c)
    tensor2c = Tensor(j, k, l, name="c", symmetry=symmetry_3c)
    tensor2d = Tensor(j, k, l, name="d", symmetry=symmetry_3c)
    output = Tensor(i, l, name="output")
    expr = (tensor1a + tensor1b * 2) * (tensor2c + tensor2d * 0.5)
    expr = expr.expand()
    assert repr(output) == "output(i,l)"
    assert repr(expr) == "(a(i,j,k) * c(j,k,l)) + (0.5 * a(i,j,k) * d(j,k,l)) + (2 * b(i,j,k) * c(j,k,l)) + (b(i,j,k) * d(j,k,l))"

    output_expr_opt = optimise_gristmill([output], [expr], strategy="exhaust")
    output_opt, expr_opt = zip(*sorted(output_expr_opt))
    expr_sub = substitute_expressions(output_expr_opt).expand()
    assert expr.canonicalise() == expr_sub.canonicalise()
