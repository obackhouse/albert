from albert import Add, Mul, Scalar, Tensor, canonical_key


def test_canonical_key_matches_independently_built_ast_nodes() -> None:
    first = Add([Tensor("A", ["i"]), Mul([Scalar(2), Tensor("B", ["i"])])])
    second = Add([Tensor("A", ["i"]), Mul([Scalar(2), Tensor("B", ["i"])])])

    assert canonical_key(first) == canonical_key(second)
    assert hash(canonical_key(first)) == hash(canonical_key(second))
    reordered = Add([Mul([Scalar(2), Tensor("B", ["i"])]), Tensor("A", ["i"])])
    assert canonical_key(first) != canonical_key(reordered)


def test_canonical_key_preserves_exact_index_labels_and_order() -> None:
    original = Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k", "j"])])
    renamed = Mul([Tensor("A", ["p", "q"]), Tensor("B", ["q", "r"])])
    different_pattern = Mul([Tensor("A", ["p", "p"]), Tensor("B", ["q", "r"])])

    assert canonical_key(original) != canonical_key(renamed)
    assert canonical_key(original) != canonical_key(different_pattern)


def test_nodes_compare_and_hash_by_structure() -> None:
    first = Add([Tensor("A", ["i"]), Tensor("B", ["i"])])
    equivalent = Add([Tensor("A", ["i"]), Tensor("B", ["i"])])
    different_indices = Add([Tensor("A", ["j"]), Tensor("B", ["j"])])
    different_order = Add([Tensor("B", ["i"]), Tensor("A", ["i"])])

    assert first == equivalent
    assert hash(first) == hash(equivalent)
    assert len({first, equivalent}) == 1
    assert first != different_indices
    assert first != different_order
    assert first != object()


def test_scalar_equality_and_hash_use_the_scalar_value() -> None:
    positive_zero = Scalar(0.0)
    negative_zero = Scalar(-0.0)
    two = Scalar(2.0)

    assert positive_zero == negative_zero
    assert hash(positive_zero) == hash(negative_zero)
    assert two == Scalar(2.0)
    assert hash(two) == hash(Scalar(2.0))
    assert two != Scalar(3.0)


def test_tensor_equality_and_hash_include_symbol_and_ordered_indices() -> None:
    tensor = Tensor("T", ["i", "j"])
    equivalent = Tensor("T", ["i", "j"])

    assert tensor == equivalent
    assert hash(tensor) == hash(equivalent)
    assert tensor != Tensor("U", ["i", "j"])
    assert tensor != Tensor("T", ["j", "i"])
    assert tensor != Tensor("T", ["i"])


def test_node_equality_and_hash_include_kind_arity_and_child_structure() -> None:
    left = Tensor("A", [])
    right = Tensor("B", [])
    sum_node = Add([left, right])
    equivalent_sum = Add([Tensor("A", []), Tensor("B", [])])
    product_node = Mul([left, right])

    assert sum_node == equivalent_sum
    assert hash(sum_node) == hash(equivalent_sum)
    assert sum_node != product_node
    assert sum_node != Add([left])
    assert sum_node != Add([right, left])
    assert sum_node != Add([left, Mul([Scalar(1), right])])


def test_equal_nodes_are_interchangeable_as_mapping_and_set_keys() -> None:
    original = Add([Tensor("A", ["i"]), Tensor("B", ["i"])])
    equivalent = Add([Tensor("A", ["i"]), Tensor("B", ["i"])])

    values = {original: "found"}

    assert values[equivalent] == "found"
    assert len({original, equivalent}) == 1


def test_empty_add_and_mul_are_distinct_hashable_nodes() -> None:
    empty_sum = Add([])
    empty_product = Mul([])
    equivalent_empty_sum = Add([])

    assert empty_sum == equivalent_empty_sum
    assert hash(empty_sum) == hash(equivalent_empty_sum)
    assert empty_sum != empty_product
    assert hash(empty_sum) == hash(canonical_key(empty_sum))


def test_repeated_index_patterns_participate_in_hashing_and_equality() -> None:
    repeated = Tensor("T", ["i", "i", "j"])
    equivalent = Tensor("T", ["i", "i", "j"])

    assert repeated == equivalent
    assert hash(repeated) == hash(equivalent)
    assert repeated != Tensor("T", ["i", "j", "j"])
    assert repeated != Tensor("T", ["i", "j"])


def test_nested_shared_and_independent_subtrees_hash_equally() -> None:
    shared = Mul([Scalar(2), Tensor("A", ["i"])])
    expression_with_shared_child = Add([shared, shared])
    expression_with_independent_children = Add(
        [Mul([Scalar(2), Tensor("A", ["i"])]), Mul([Scalar(2), Tensor("A", ["i"])])]
    )

    assert expression_with_shared_child == expression_with_independent_children
    assert hash(expression_with_shared_child) == hash(expression_with_independent_children)


def test_nesting_and_child_counts_are_part_of_the_key() -> None:
    a = Tensor("A", ["i"])
    b = Tensor("B", ["i"])
    nested = Add([Add([a, b]), a])
    flat = Add([a, b, a])
    shorter = Add([Add([a, b])])

    assert nested != flat
    assert nested != shorter
    assert hash(nested) == hash(canonical_key(nested))


def test_string_contents_are_hashed_exactly() -> None:
    original = Tensor("T\0ensor", ["ί", "j"])
    equivalent = Tensor("T\0ensor", ["ί", "j"])

    assert original == equivalent
    assert hash(original) == hash(equivalent)
    assert original != Tensor("Tensor", ["ί", "j"])
    assert original != Tensor("T\0ensor", ["i", "j"])
