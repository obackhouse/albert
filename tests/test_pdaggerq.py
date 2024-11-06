from typing import Callable
import pytest
from albert.algebra import Mul
from albert.base import Base
from albert.index import Index
from albert.scalar import Scalar
from albert.tensor import Tensor
from albert.qc._pdaggerq import import_from_pdaggerq, remove_reference_energy

try:
    import pdaggerq
except:
    pdaggerq = None



@pytest.mark.skipif(pdaggerq is None, reason="pdaggerq is not installed")
def test_ccsd_energy():
    pq = pdaggerq.pq_helper("fermi")
    pq.set_print_level(0)
    pq.set_left_operators([["1"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    terms = pq.fully_contracted_strings()
    expr_ghf = import_from_pdaggerq(terms)
    expr_ghf = expr_ghf.canonicalise()
    assert repr(expr_ghf) == "(-0.5 * v(j,i,j,i)) + f(i,i) + (0.25 * t(i,j,a,b) * v(i,j,a,b)) + (f(i,a) * t(i,a)) + (0.5 * t(i,a) * t(j,b) * v(i,j,a,b))"
    assert all(i.spin == None for i in expr_ghf.external_indices)
    assert all(i.spin == None for i in expr_ghf.internal_indices)

    terms = remove_reference_energy(terms)
    expr_ghf = import_from_pdaggerq(terms)
    expr_ghf = expr_ghf.canonicalise()
    assert repr(expr_ghf) == "(0.25 * t(i,j,a,b) * v(i,j,a,b)) + (f(i,a) * t(i,a)) + (0.5 * t(i,a) * t(j,b) * v(i,j,a,b))"

    def _filter_fock_terms(mul: Mul) -> Mul | Scalar:
        """Filter out off-diagonal Fock terms."""
        for child in mul._children:
            if isinstance(child, Tensor) and child.name == "f" and child.indices[0].space != child.indices[1].space:
                return Scalar(0.0)
        return mul

    expr_ghf = expr_ghf.expand()
    expr_ghf = expr_ghf.apply(_filter_fock_terms, Mul)
    expr_ghf = expr_ghf.canonicalise()
    assert repr(expr_ghf) == "(0.25 * t(i,j,a,b) * v(i,j,a,b)) + (0.5 * t(i,a) * t(j,b) * v(i,j,a,b))"

    def _project_onto_indices(indices: tuple[Index, ...]) -> Callable[[Base], Base]:
        """Get a function to project onto the indices."""

        def _project(mul: Mul) -> Mul | Scalar:
            """Project onto the indices."""
            mul_indices = mul.external_indices + mul.internal_indices
            for index in indices:
                if index not in mul_indices:
                    return Scalar(0.0)
            return mul

        return _project

    expr_uhf = expr_ghf.as_uhf()[0]

    expr_uhf_aaaa = expr_uhf.apply(
        _project_onto_indices(
            (
                Index("i", space="o", spin="a"),
                Index("j", space="o", spin="a"),
                Index("a", space="v", spin="a"),
                Index("b", space="v", spin="a"),
            )
        ),
        Mul,
    )
    expr_uhf_aaaa = expr_uhf_aaaa.canonicalise()
    assert repr(expr_uhf_aaaa) == "(0.5 * t(iα,aα) * t(jα,bα) * v(iα,aα,jα,bα)) + (-0.5 * t(iα,aα) * t(jα,bα) * v(iα,bα,jα,aα))"

    expr_uhf_abab = expr_uhf.apply(
        _project_onto_indices(
            (
                Index("i", space="o", spin="a"),
                Index("j", space="o", spin="b"),
                Index("a", space="v", spin="a"),
                Index("b", space="v", spin="b"),
            )
        ),
        Mul,
    )
    expr_uhf_abab = expr_uhf_abab.canonicalise()
    assert repr(expr_uhf_abab) == "(0.25 * t(iα,jβ,aα,bβ) * v(iα,aα,jβ,bβ)) + (0.5 * t(iα,aα) * t(jβ,bβ) * v(iα,aα,jβ,bβ))"