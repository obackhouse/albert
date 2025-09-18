from albert.algebra import Mul
from albert.canon import canonicalise_indices
from albert.index import Index
from albert.qc.ghf import ERI
from albert.qc.spin import ghf_to_rhf
from albert.scalar import Scalar


def test_ghf_to_rhf():
    i = Index("i", space="o", spin=None)
    j = Index("j", space="o", spin=None)
    a = Index("a", space="v", spin=None)
    b = Index("b", space="v", spin=None)
    ijab = ERI(i, j, a, b)
    expr_ghf = Mul(ijab, ijab) * 0.25
    assert expr_ghf == expr_ghf.canonicalise()
    assert expr_ghf._children == (Scalar(0.25), ijab, ijab)
    assert expr_ghf == ijab * ijab * 0.25
    assert expr_ghf == 0.25 * ijab * ijab
    assert expr_ghf.external_indices == ()
    assert expr_ghf.internal_indices == (i, j, a, b)
    assert not expr_ghf.disjoint
    assert repr(expr_ghf) == "0.25 * v(i,j,a,b) * v(i,j,a,b)"

    # Project onto (ab|ab)
    expr_ghf = expr_ghf.map_indices(
        dict(
            i=i.copy(spin="a"),
            j=j.copy(spin="b"),
            a=a.copy(spin="a"),
            b=b.copy(spin="b"),
        ),
    )

    expr_rhf = ghf_to_rhf(expr_ghf)  # All have no external indices
    expr_rhf = canonicalise_indices(expr_rhf.expand().canonicalise()).collect()
    assert expr_rhf == expr_rhf.canonicalise()
    assert repr(expr_rhf) == "(2 * v(i,a,j,b) * v(i,a,j,b)) + (-1 * v(i,a,j,b) * v(i,b,j,a))"
    assert all(i.spin == "r" for i in expr_rhf.internal_indices)
    assert all(i.spin == "r" for i in expr_rhf.external_indices)
