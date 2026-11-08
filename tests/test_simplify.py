import pytest

from albert import Add, Mul, Scalar, Tensor, collect, expand, flatten, prune, simplify, squeeze


def _product_factors(term):
    assert isinstance(term, Mul)
    return term.children


def test_expand_distributes_a_product_over_a_sum() -> None:
    a = Tensor("A", ["i"])
    b = Tensor("B", ["i"])
    c = Tensor("C", [])

    result = expand(Mul([Add([a, b]), c]))

    assert isinstance(result, Add)
    assert len(result.children) == 2
    assert _product_factors(result.children[0]) == [a, c]
    assert _product_factors(result.children[1]) == [b, c]


def test_expand_preserves_ranked_indices_and_internal_factors() -> None:
    a = Tensor("A", ["i", "j"])
    b = Tensor("B", ["i", "k"])
    c = Tensor("C", ["k", "j"])
    d = Tensor("D", ["l"])
    expression = Mul([Add([a, Mul([b, c])]), d])

    result = expand(expression)

    assert isinstance(result, Add)
    assert len(result.children) == 2
    first_term = _product_factors(result.children[0])
    second_term = _product_factors(result.children[1])
    assert first_term == [a, d]
    assert second_term == [b, c, d]
    assert all(factor.rank > 0 for factor in first_term + second_term)


def test_expand_flattens_nested_sums_and_products() -> None:
    a = Tensor("A", [])
    b = Tensor("B", [])
    c = Tensor("C", [])
    expression = Mul([Add([a, Add([b, c])]), Scalar(2)])

    result = expand(expression)

    assert isinstance(result, Add)
    assert len(result.children) == 3
    assert [_product_factors(term)[-1].value for term in result.children] == [2, 2, 2]
    assert all(factor.kind in {a.kind, Scalar(0).kind} for term in result.children for factor in _product_factors(term))


def test_expand_of_a_leaf_has_uniform_add_mul_depth() -> None:
    tensor = Tensor("A", ["i"])

    result = expand(tensor)

    assert isinstance(result, Add)
    assert len(result.children) == 1
    assert _product_factors(result.children[0]) == [tensor]


def test_collect_combines_duplicate_terms_and_coefficients() -> None:
    tensor = Tensor("A", [])
    expression = Add([tensor, tensor, Mul([Scalar(3), tensor]), Scalar(2), Scalar(4)])

    result = collect(expression)

    assert isinstance(result, Add)
    assert len(result.children) == 2
    assert isinstance(result.children[0], Mul)
    assert result.children[0].children[0].value == 5
    assert result.children[0].children[1] is tensor
    assert isinstance(result.children[1], Scalar)
    assert result.children[1].value == 6


def test_collect_combines_like_ranked_tensor_terms() -> None:
    tensor = Tensor("T", ["i", "j"])
    expression = Add([tensor, Mul([Scalar(2), tensor]), tensor])

    result = collect(expression)

    assert isinstance(result, Add)
    assert len(result.children) == 1
    assert isinstance(result.children[0], Mul)
    assert result.children[0].children[0].value == 4
    assert result.children[0].children[1] is tensor
    assert result.children[0].children[1].indices == ["i", "j"]


def test_collect_keeps_reversed_indices_distinct_without_symmetries() -> None:
    ij = Tensor("T", ["i", "j"])
    ji = Tensor("T", ["j", "i"])
    expression = Add([ij, ji])

    result = collect(expression)

    assert isinstance(result, Add)
    assert result.children == [ij, ji]


def test_collect_does_not_change_non_add_nodes() -> None:
    product = Mul([Tensor("A", []), Tensor("B", [])])

    assert collect(product) is product


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (Add([Tensor("A", [])]), Tensor("A", [])),
        (Mul([Tensor("A", [])]), Tensor("A", [])),
    ],
)
def test_squeeze_returns_the_only_child(node, expected) -> None:
    result = squeeze(node)

    assert isinstance(result, Tensor)
    assert result.symbol == expected.symbol


def test_squeeze_preserves_nodes_with_other_arities() -> None:
    node = Add([Tensor("A", []), Tensor("B", [])])

    assert squeeze(node) is node


def test_flatten_only_flattens_nested_products() -> None:
    a = Tensor("A", ["i"])
    b = Tensor("B", ["j"])
    c = Tensor("C", ["k"])
    d = Tensor("D", ["i"])
    e = Tensor("E", ["j"])
    f = Tensor("F", ["k"])
    sum_node = Add([Mul([a, Mul([b, c])]), Mul([d, Mul([e, f])])])

    result = flatten(sum_node)

    assert isinstance(result, Add)
    assert result.children[0] == Mul([a, b, c])
    assert result.children[1] == Mul([d, e, f])


def test_flatten_preserves_additions_inside_products() -> None:
    a = Tensor("A", ["i"])
    b = Tensor("B", ["i"])
    c = Tensor("C", ["i"])
    expression = Mul([a, Mul([Add([b, c]), Scalar(2)])])

    result = flatten(expression)

    assert isinstance(result, Mul)
    assert result.children == [a, Add([b, c]), Scalar(2)]


def test_prune_removes_neutral_elements_from_add_and_mul() -> None:
    tensor_a = Tensor("A", [])
    tensor_b = Tensor("B", [])

    sum_result = prune(Add([Scalar(0), tensor_a]))
    product_result = prune(Mul([Scalar(1), tensor_b]))

    assert isinstance(sum_result, Add)
    assert sum_result.children == [tensor_a]
    assert isinstance(product_result, Mul)
    assert product_result.children == [tensor_b]


def test_remove_all_neutral_elements_returns_the_identity_scalar() -> None:
    sum_result = prune(Add([Scalar(0), Scalar(0)]))
    product_result = prune(Mul([Scalar(1), Scalar(1)]))

    assert isinstance(sum_result, Scalar)
    assert sum_result.value == 0
    assert isinstance(product_result, Scalar)
    assert product_result.value == 1


def test_simplify_collects_removes_identities_and_squeezes() -> None:
    tensor = Tensor("A", [])
    expression = Add([Scalar(0), Mul([Scalar(1), tensor]), tensor])

    result = simplify(expression)

    assert isinstance(result, Mul)
    assert len(result.children) == 2
    assert result.children[0].value == 2
    assert result.children[1] is tensor


def test_simplify_expands_small_products_of_sums() -> None:
    a = Tensor("A", [])
    b = Tensor("B", [])
    c = Tensor("C", [])
    expression = Mul([Add([a, b]), c])

    result = simplify(expression)

    assert isinstance(result, Add)
    assert len(result.children) == 2
    assert _product_factors(result.children[0]) == [a, c]
    assert _product_factors(result.children[1]) == [b, c]


def test_simplify_expands_indexed_sums_while_preserving_external_indices() -> None:
    a = Tensor("A", ["i"])
    b = Tensor("B", ["i"])
    c = Tensor("C", ["j"])

    result = simplify(Mul([Add([a, b]), c]))

    assert isinstance(result, Add)
    assert len(result.children) == 2
    first_term, second_term = map(_product_factors, result.children)
    assert first_term == [a, c]
    assert second_term == [b, c]
    assert [factor.indices for factor in first_term] == [["i"], ["j"]]


def test_simplify_handles_a_product_with_internal_indices() -> None:
    a = Tensor("A", ["i", "k"])
    b = Tensor("B", ["k", "j"])
    expression = Mul([Scalar(1), a, b])

    result = simplify(expression)

    assert isinstance(result, Mul)
    assert result.children == [a, b]


def test_simplify_collects_terms_across_a_larger_distributive_expression() -> None:
    a = Tensor("A", ["i"])
    b = Tensor("B", ["i"])
    c = Tensor("C", ["j"])
    d = Tensor("D", ["k"])
    e = Tensor("E", ["k"])
    expression = Mul([Add([a, a, b]), Add([c, c]), Add([d, e])])

    result = simplify(expression)

    assert isinstance(result, Add)
    terms = {}
    for term in result.children:
        assert isinstance(term, Mul)
        coefficient, *factors = term.children
        assert isinstance(coefficient, Scalar)
        assert all(isinstance(factor, Tensor) for factor in factors)
        terms[tuple(factor.symbol for factor in factors)] = coefficient.value
    assert terms == {
        ("A", "C", "D"): 4,
        ("A", "C", "E"): 4,
        ("B", "C", "D"): 2,
        ("B", "C", "E"): 2,
    }


def test_simplify_keeps_products_above_expansion_limit_factored() -> None:
    sums = [Add([Tensor(f"A{index}", [f"i{index}"]), Tensor(f"B{index}", [f"i{index}"])]) for index in range(9)]
    expression = Mul(sums)

    result = simplify(expression)

    assert isinstance(result, Mul)
    assert len(result.children) == 9
    assert all(isinstance(child, Add) for child in result.children)
