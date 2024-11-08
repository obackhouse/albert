
from albert.index import Index


def test_index():
    index = Index("i")
    assert index.name == "i"
    assert index.spin is None
    assert index.space is None

    index_flip = index.spin_flip()
    assert index_flip.name == "i"
    assert index_flip.spin == None
    assert index_flip.space == None

    assert index == index_flip
    assert hash(index) == hash(index_flip)
    assert hash(index) == index._hash
    assert index._hashable() == (Index.__name__, "", "", "i")
    assert repr(index) == "i"

    json = index.as_json()
    index_from_json = Index.from_json(json)
    assert index == index_from_json
    assert hash(index) == hash(index_from_json)
    assert hash(index) == index_from_json._hash


def test_index_spin():
    index = Index("i", spin="a")
    assert index.name == "i"
    assert index.spin == "a"
    assert index.space is None

    index_flip = index.spin_flip()
    assert index_flip.name == "i"
    assert index_flip.spin == "b"
    assert index_flip.space == None

    assert index != index_flip
    assert index < index_flip
    assert index._hash is None  # Will be set after first call
    assert hash(index) != hash(index_flip)
    assert index._hash is not None
    assert hash(index) == index._hash
    assert index._hashable() == (Index.__name__, "", "a", "i")
    assert repr(index) == "iα"

    json = index.as_json()
    index_from_json = Index.from_json(json)
    assert index == index_from_json
    assert hash(index) == hash(index_from_json)
    assert hash(index) == index_from_json._hash


def test_index_space():
    index = Index("i", space="o")
    assert index.name == "i"
    assert index.spin is None
    assert index.space == "o"

    index_flip = index.spin_flip()
    assert index_flip.name == "i"
    assert index_flip.spin == None
    assert index_flip.space == "o"

    assert index == index_flip
    assert index._hash is None  # Will be set after first call
    assert hash(index) == hash(index_flip)
    assert index._hash is not None
    assert hash(index) == index._hash
    assert index._hashable() == (Index.__name__, "o", "", "i")
    assert repr(index) == "i"

    json = index.as_json()
    index_from_json = Index.from_json(json)
    assert index == index_from_json
    assert hash(index) == hash(index_from_json)
    assert hash(index) == index_from_json._hash


def test_index_spin_space():
    index = Index("i", spin="a", space="o")
    assert index.name == "i"
    assert index.spin == "a"
    assert index.space == "o"

    index_flip = index.spin_flip()
    assert index_flip.name == "i"
    assert index_flip.spin == "b"
    assert index_flip.space == "o"

    assert index != index_flip
    assert index < index_flip
    assert index._hash is None  # Will be set after first call
    assert hash(index) != hash(index_flip)
    assert index._hash is not None
    assert hash(index) == index._hash
    assert index._hashable() == (Index.__name__, "o", "a", "i")
    assert repr(index) == "iα"

    json = index.as_json()
    index_from_json = Index.from_json(json)
    assert index == index_from_json
    assert hash(index) == hash(index_from_json)
    assert hash(index) == index_from_json._hash
