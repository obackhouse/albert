from albert import Add, Mul, Tensor


def test_binary_operators_create_binary_nodes_without_flattening() -> None:
    left = Tensor("A", ["i"])
    middle = Tensor("B", ["i"])
    right = Tensor("C", ["i"])

    sum_node = left + middle
    product_node = left * middle
    nested_sum = sum_node + right
    nested_product = product_node * right

    assert isinstance(sum_node, Add)
    assert sum_node.arity == 2
    assert isinstance(product_node, Mul)
    assert product_node.arity == 2
    assert isinstance(nested_sum, Add)
    assert nested_sum.arity == 2
    assert isinstance(nested_sum.children[0], Add)
    assert isinstance(nested_product, Mul)
    assert nested_product.arity == 2
    assert isinstance(nested_product.children[0], Mul)


def test_reflected_numeric_operators_preserve_operand_order() -> None:
    value = Tensor("A", [])

    added = 2.0 + value
    multiplied = 3.0 * value
    subtracted = 5.0 - value
    right_added = value + 2.0
    right_multiplied = value * 3.0
    right_subtracted = value - 4.0

    assert isinstance(added, Add)
    assert added.children[0].value == 2.0
    assert isinstance(added.children[1], Tensor)
    assert isinstance(multiplied, Mul)
    assert multiplied.children[0].value == 3.0
    assert isinstance(multiplied.children[1], Tensor)
    assert isinstance(subtracted, Add)
    assert subtracted.children[0].value == 5.0
    assert isinstance(subtracted.children[1], Mul)
    assert subtracted.children[1].children[0].value == -1.0
    assert isinstance(right_added, Add)
    assert isinstance(right_added.children[0], Tensor)
    assert right_added.children[1].value == 2.0
    assert isinstance(right_multiplied, Mul)
    assert isinstance(right_multiplied.children[0], Tensor)
    assert right_multiplied.children[1].value == 3.0
    assert isinstance(right_subtracted, Add)
    assert isinstance(right_subtracted.children[0], Tensor)
    assert isinstance(right_subtracted.children[1], Mul)
    assert right_subtracted.children[1].children[1].value == 4.0


def test_subtraction_and_unary_signs_build_explicit_nodes() -> None:
    left = Tensor("A", [])
    right = Tensor("B", [])

    difference = left - right
    negative = -left
    positive = +left

    assert isinstance(difference, Add)
    assert difference.arity == 2
    assert isinstance(difference.children[1], Mul)
    assert difference.children[1].children[0].value == -1.0
    assert isinstance(negative, Mul)
    assert negative.children[0].value == -1.0
    assert isinstance(positive, Mul)
    assert positive.children[0].value == 1.0
