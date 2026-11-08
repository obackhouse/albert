from albert import Add, Mul, Scalar, Tensor, simplify


def test_indexed_tensor_expression_expands_and_matches_expected_ast() -> None:
    a = Tensor("A", ["i", "k"])
    b = Tensor("B", ["k", "j"])
    c = Tensor("C", ["i", "j"])
    d = Tensor("D", ["l"])
    e = Tensor("E", ["l"])
    expression = (a * b + c) * (d + e)

    result = simplify(expression)
    expected = Add([Mul([a, b, d]), Mul([a, b, e]), Mul([c, d]), Mul([c, e])])

    assert result == expected
    assert hash(result) == hash(expected)
    assert len(result.search(lambda node: isinstance(node, Tensor))) == 10
    assert all(isinstance(term, Mul) for term in result.children)


def test_simplify_collects_coefficients_after_distributing_indexed_products() -> None:
    a = Tensor("A", ["i", "k"])
    b = Tensor("B", ["k", "j"])
    c = Tensor("C", ["l"])
    expression = (2 * (a * b) + a * b) * (c + c)

    result = simplify(expression)
    expected = Mul([Scalar(6), a, b, c])

    assert result == expected
    assert hash(result) == hash(expected)
    assert len(result.search(lambda node: isinstance(node, Tensor))) == 3
    assert result.find(lambda node: isinstance(node, Add)) is None


def test_large_indexed_product_expands_to_all_tensor_contraction_paths() -> None:
    x_terms = [
        Mul([Tensor("A", ["i", "x"]), Tensor("B", ["x", "j"])]),
        Mul([Tensor("C", ["i", "y"]), Tensor("D", ["y", "j"])]),
        Mul([Tensor("E", ["i", "z"]), Tensor("F", ["z", "j"])]),
        Tensor("G", ["i", "j"]),
    ]
    y_terms = [
        Mul([Tensor("H", ["j", "u"]), Tensor("I", ["u", "k"])]),
        Mul([Tensor("J", ["j", "v"]), Tensor("K", ["v", "k"])]),
        Tensor("L", ["j", "k"]),
        Mul([Tensor("M", ["j", "w"]), Tensor("N", ["w", "k"])]),
    ]
    z_terms = [
        Mul([Tensor("O", ["k", "p"]), Tensor("P", ["p", "l"])]),
        Mul([Tensor("Q", ["k", "r"]), Tensor("S", ["r", "l"])]),
        Tensor("T", ["k", "l"]),
        Mul([Tensor("U", ["k", "s"]), Tensor("V", ["s", "l"])]),
    ]
    expression = Mul([Add(x_terms), Add(y_terms), Add(z_terms)])

    result = simplify(expression)

    def symbols(term):
        if isinstance(term, Tensor):
            return (term.symbol,)
        return tuple(factor.symbol for factor in term.children)

    expected_paths = {
        symbols(x_term) + symbols(y_term) + symbols(z_term)
        for x_term in x_terms
        for y_term in y_terms
        for z_term in z_terms
    }
    actual_paths = {tuple(factor.symbol for factor in term.children) for term in result.children}
    equivalent_result = Add([Mul(list(term.children)) for term in result.children])

    assert isinstance(result, Add)
    assert len(result.children) == 64
    assert actual_paths == expected_paths
    assert len(result.search(lambda node: isinstance(node, Tensor))) == 336
    assert all(isinstance(term, Mul) for term in result.children)
    assert result == equivalent_result
    assert hash(result) == hash(equivalent_result)


def test_larger_repeated_sums_collect_all_coefficients_after_expansion() -> None:
    a = Tensor("A", ["i"])
    b = Tensor("B", ["i"])
    c = Tensor("C", ["i"])
    d = Tensor("D", ["j"])
    e = Tensor("E", ["j"])
    expression = Mul([Add([a, a, b, c]), Add([d, d, e])])

    result = simplify(expression)

    terms = {}
    for term in result.children:
        assert isinstance(term, Mul)
        factors = term.children
        coefficient = 1
        if isinstance(factors[0], Scalar):
            coefficient = factors[0].value
            factors = factors[1:]
        terms[tuple(factor.symbol for factor in factors)] = coefficient

    assert isinstance(result, Add)
    assert terms == {
        ("A", "D"): 4,
        ("A", "E"): 2,
        ("B", "D"): 2,
        ("B", "E"): 1,
        ("C", "D"): 2,
        ("C", "E"): 1,
    }
