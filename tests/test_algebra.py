import pytest

from albert.algebra import Add, Mul
from albert.index import Index
from albert.scalar import Scalar
from albert.tensor import Tensor


def test_add():
    i = Index("i")
    j = Index("j")
    tensor1 = Tensor(i, j, name="m")
    tensor2 = Tensor(i, j, name="n")
    add = Add(tensor1, tensor2)
    assert add._children == (tensor1, tensor2)
    assert add == tensor1 + tensor2
    assert add != tensor2 + tensor1
    assert add == (tensor2 + tensor1).canonicalise()
    assert add.external_indices == (i, j)
    assert add.internal_indices == ()
    assert not add.disjoint
    assert repr(add) == "m(i,j) + n(i,j)"
    assert add == Tensor.from_string(repr(add))

    add_copy = add.copy()
    assert add == add_copy
    assert add._hash is None  # Will be set after first call
    assert hash(add) == hash(add_copy)
    assert add._hash is not None
    assert hash(add) == add._hash

    assert tensor1 + Scalar(0.0) == tensor1
    assert Scalar(0.0) + tensor1 == tensor1
    assert Scalar(0.0) + Scalar(0.0) == Scalar(0.0)
    assert Scalar(0.0) + Scalar(1.0) == Scalar(1.0)
    assert Scalar(1.0) + Scalar(0.0) == Scalar(1.0)
    assert Scalar(1.0) + Scalar(1.0) == Scalar(2.0)
    assert Scalar(2.0) + Scalar(3.0) == Scalar(5.0)

    with pytest.raises(ValueError):
        Add(tensor1, Scalar(1.0))
        tensor1 + Scalar(1.0)

    tensor3 = Tensor(i, name="o")

    with pytest.raises(ValueError):
        Add(tensor1, tensor3)
        tensor1 + tensor3


def test_mul():
    i = Index("i")
    j = Index("j")
    tensor1 = Tensor(i, j, name="m")
    tensor2 = Tensor(i, j, name="n")
    mul = Mul(tensor1, tensor2)
    assert mul._children == (tensor1, tensor2)
    assert mul == tensor1 * tensor2
    assert mul != tensor2 * tensor1
    assert mul == (tensor2 * tensor1).canonicalise()
    assert mul.external_indices == ()  # Einstein summation
    assert mul.internal_indices == (i, j)
    assert not mul.disjoint
    assert repr(mul) == "m(i,j) * n(i,j)"
    assert mul == Tensor.from_string(repr(mul))

    mul_copy = mul.copy()
    assert mul == mul_copy
    assert mul._hash is None  # Will be set after first call
    assert hash(mul) == hash(mul_copy)
    assert mul._hash is not None
    assert hash(mul) == mul._hash

    assert tensor1 * Scalar(1.0) == tensor1
    assert Scalar(1.0) * tensor1 == tensor1
    assert Scalar(0.0) * tensor1 == Scalar(0.0)
    assert Scalar(0.0) * Scalar(0.0) == Scalar(0.0)
    assert Scalar(0.0) * Scalar(1.0) == Scalar(0.0)
    assert Scalar(1.0) * Scalar(0.0) == Scalar(0.0)
    assert Scalar(1.0) * Scalar(1.0) == Scalar(1.0)
    assert Scalar(2.0) * Scalar(3.0) == Scalar(6.0)

    tensor3 = Tensor(i, name="o")
    assert (tensor1 * tensor3).external_indices == (j,)

    with pytest.raises(ValueError):
        tensor3 * tensor3 * tensor1

    k = Index("k")
    tensor4 = Tensor(j, k, name="p")
    mul2 = Mul(tensor1, tensor4)
    assert mul2.external_indices == (i, k)
    assert mul2.internal_indices == (j,)
    assert repr(mul2) == "m(i,j) * p(j,k)"
    assert mul2 == Tensor.from_string(repr(mul2))
    assert mul2 == tensor1 * tensor4
    assert mul2 != tensor4 * tensor1
    assert mul2 == (tensor4 * tensor1).canonicalise(indices=False)
    assert repr((tensor4 * tensor1).canonicalise(indices=False)) == "m(i,j) * p(j,k)"
    assert mul2 != (tensor4 * tensor1).canonicalise(indices=True)
    assert repr((tensor4 * tensor1).canonicalise(indices=True)) == "m(i,k) * p(k,j)"


def test_nested():
    i = Index("i")
    j = Index("j")
    k = Index("k")
    l = Index("l")
    tensor1 = Tensor(i, j, name="t1")
    tensor2 = Tensor(i, j, name="t2")
    tensor3 = Tensor(j, k, name="t3")
    tensor4 = Tensor(k, l, name="t4")
    expr = (tensor1 + tensor2) * tensor3 * tensor4
    assert expr.external_indices == (i, l)
    assert expr.internal_indices == (j, k)
    assert not expr.disjoint
    assert repr(expr) == "(t1(i,j) + t2(i,j)) * t3(j,k) * t4(k,l)"
    assert expr == Mul(Add(tensor1, tensor2), tensor3, tensor4)
    assert expr != (tensor2 + tensor1) * tensor4 * tensor3
    assert expr == ((tensor2 + tensor1) * tensor4 * tensor3).canonicalise(indices=False)
    assert (
        repr(((tensor2 + tensor1) * tensor4 * tensor3).canonicalise(indices=False))
        == "(t1(i,j) + t2(i,j)) * t3(j,k) * t4(k,l)"
    )
    assert expr != ((tensor2 + tensor1) * tensor4 * tensor3).canonicalise(indices=True)
    assert (
        repr(((tensor2 + tensor1) * tensor4 * tensor3).canonicalise(indices=True))
        == "(t1(i,k) * t3(k,l) * t4(l,j)) + (t2(i,k) * t3(k,l) * t4(l,j))"
    )  # index canonicalisation expands the expression

    expanded = expr.expand()
    assert isinstance(expanded, Add)
    assert all(isinstance(child, Mul) for child in expanded._children)
    assert expanded.external_indices == (i, l)
    assert expanded.internal_indices == ()
    assert repr(expanded) == "(t1(i,j) * t3(j,k) * t4(k,l)) + (t2(i,j) * t3(j,k) * t4(k,l))"
    assert expanded == Add(Mul(tensor1, tensor3, tensor4), Mul(tensor2, tensor3, tensor4))
