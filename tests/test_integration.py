from itertools import product

from albert import Add, Mul, Scalar, Tensor, canonicalize, simplify


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


def test_large_indexed_product_expands_to_all_internal_index_paths() -> None:
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


def test_eight_layer_indexed_sum_product_expands_and_collects_long_expression() -> None:
    layer_sums = []
    for layer in range(8):
        left = Tensor(f"A{layer}", [f"i{layer}", f"i{layer + 1}"])
        right = Tensor(f"B{layer}", [f"i{layer}", f"i{layer + 1}"])
        layer_sums.append(Add([left, right]))

    result = simplify(Mul(layer_sums))

    assert isinstance(result, Add)
    assert len(result.children) == 256
    assert result.external_indices == ["i0", "i8"]
    assert len(result.search(lambda node: isinstance(node, Tensor))) == 2048
    assert all(isinstance(term, Mul) and len(term.children) == 8 for term in result.children)
    actual_paths = {tuple(factor.symbol for factor in term.children) for term in result.children}
    expected_paths = {
        tuple(f"{'AB'[choice]}{layer}" for layer, choice in enumerate(choices)) for choices in product((0, 1), repeat=8)
    }
    assert actual_paths == expected_paths


def test_long_sum_of_contractions_canonicalizes_factor_order_and_dummy_names() -> None:
    terms = []
    renamed_terms = []
    for term_number in range(32):
        factors = []
        renamed_factors = []
        for pair_number in range(4):
            symbol = f"T{term_number}_{pair_number}"
            index = f"d{term_number}_{pair_number}"
            renamed_index = f"x{term_number}_{pair_number}"
            factors.extend([Tensor(f"{symbol}a", [index]), Tensor(f"{symbol}b", [index])])
            renamed_factors.extend([Tensor(f"{symbol}b", [renamed_index]), Tensor(f"{symbol}a", [renamed_index])])

        terms.append(Mul(factors))
        renamed_terms.append(Mul(renamed_factors))

    expression = Add(terms)
    renamed_expression = Add(renamed_terms)

    canonical = canonicalize(expression)

    assert len(canonical.children) == 32
    assert len(canonical.search(lambda node: isinstance(node, Tensor))) == 256
    assert canonical == canonicalize(renamed_expression)
