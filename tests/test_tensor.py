import pytest
from albert.index import Index
from albert.scalar import Scalar
from albert.tensor import Tensor
from albert.symmetry import Permutation, Symmetry
from albert.algebra import Add, Mul
from albert.base import Base


def test_tensor():
    i = Index("i")
    j = Index("j")
    tensor = Tensor(i, j, name="m")
    assert tensor.indices == (i, j)
    assert tensor.name == "m"
    assert tensor.symmetry is None
    assert tensor.external_indices == (i, j)
    assert tensor.internal_indices == ()
    assert not tensor.disjoint
    assert repr(tensor) == "m(i,j)"

    tensor_copy = tensor.copy()
    assert tensor == tensor_copy
    assert tensor._hash is None  # Will be set after first call
    assert hash(tensor) == hash(tensor_copy)
    assert tensor._hash is not None
    assert hash(tensor) == tensor._hash

    tensor_copy = tensor.copy(j, i)
    tensor_mapped = tensor.map_indices({i: j, j: i})
    tensor_permed = tensor.permute_indices((1, 0))
    assert tensor_mapped == tensor_permed
    assert tensor.apply(lambda x: x, Base) == tensor
    assert tensor.canonicalise() == tensor
    assert tensor_mapped.canonicalise() == tensor_mapped  # No symmetry
    assert tensor_permed.canonicalise() == tensor_permed  # No symmetry
    assert tensor.expand() == Add(Mul(tensor))

    json = tensor.as_json()
    tensor_from_json = Tensor.from_json(json)
    assert tensor == tensor_from_json
    assert hash(tensor) == hash(tensor_from_json)
    assert hash(tensor) == tensor_from_json._hash

    assert (tensor * 1) == tensor
    assert (tensor * 0) == Scalar(0.0)


def test_tensor_symmetry():
    i = Index("i")
    j = Index("j")
    symmetry = Symmetry(Permutation((0, 1), 1), Permutation((1, 0), 1))
    tensor = Tensor(i, j, name="m", symmetry=symmetry)
    assert tensor.symmetry == symmetry
    assert tensor.external_indices == (i, j)
    assert tensor.internal_indices == ()
    assert not tensor.disjoint
    assert repr(tensor) == "m(i,j)"

    tensor_mapped = tensor.map_indices({i: j, j: i})
    tensor_permed = tensor.permute_indices((1, 0))
    assert tensor_mapped != tensor
    assert tensor_permed != tensor
    assert tensor_mapped.canonicalise() == tensor
    assert tensor_permed.canonicalise() == tensor


def test_tensor_antisymmetry():
    i = Index("i")
    j = Index("j")
    symmetry = Symmetry(Permutation((0, 1), 1), Permutation((1, 0), -1))
    tensor = Tensor(i, j, name="m", symmetry=symmetry)
    assert tensor.symmetry == symmetry
    assert repr(tensor) == "m(i,j)"

    tensor_mapped = tensor.map_indices({i: j, j: i})
    tensor_permed = tensor.permute_indices((1, 0))
    assert tensor_mapped != tensor
    assert tensor_permed != tensor
    assert repr(tensor_mapped) == "m(j,i)"
    assert repr(tensor_permed) == "m(j,i)"
    assert tensor_mapped.canonicalise() == -tensor
    assert tensor_permed.canonicalise() == -tensor
