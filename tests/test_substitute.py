import pytest

from albert import Add, Expression, Metadata, Mul, Tensor, substitute_expressions


def metadata() -> Metadata:
    return Metadata(
        indices={"I": ["i", "j", "l"], "J": ["k", "q", "r"]},
        sizes={"I": 8, "J": 8},
    )


def test_substitutes_dependencies_recursively_and_removes_definitions() -> None:
    first = Expression(Tensor("T", ["i"]), Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k"])]))
    second = Expression(Tensor("U", ["j"]), Add([Tensor("T", ["j"]), Tensor("D", ["j"])]))
    output = Expression(Tensor("Y", ["l", "q"]), Mul([Tensor("U", ["l"]), Tensor("C", ["q"])]))

    result = substitute_expressions([first, second, output], metadata())

    assert [expression.lhs.symbol for expression in result] == ["Y"]
    assert {tensor.symbol for tensor in result[0].rhs.search(lambda node: isinstance(node, Tensor))} == {
        "A",
        "B",
        "C",
        "D",
    }


def test_maps_each_reference_to_its_own_indices_and_avoids_capture() -> None:
    definition = Expression(
        Tensor("T", ["i"]),
        Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k"])]),
    )
    first = Expression(Tensor("X", ["k"]), Mul([Tensor("T", ["j"]), Tensor("C", ["j", "k"])]))
    second = Expression(Tensor("Y", []), Mul([Tensor("T", ["l"]), Tensor("D", ["l"])]))

    meta = Metadata(
        indices={"I": ["i", "j", "l"], "K": ["k", "q", "r"]},
        sizes={"I": 8, "K": 8},
    )
    result = substitute_expressions([definition, first, second], meta)
    by_symbol = {expression.lhs.symbol: expression for expression in result}
    x_tensors = by_symbol["X"].rhs.search(lambda node: isinstance(node, Tensor))
    a_tensor = next(tensor for tensor in x_tensors if tensor.symbol == "A")
    b_tensor = next(tensor for tensor in x_tensors if tensor.symbol == "B")

    assert a_tensor.indices[0] == "j"
    assert a_tensor.indices[1] == "q"
    assert b_tensor.indices == [a_tensor.indices[1]]
    assert by_symbol["X"].rhs.external_indices == ["k"]
    assert (
        next(
            tensor
            for tensor in by_symbol["Y"].rhs.search(lambda node: isinstance(node, Tensor))
            if tensor.symbol == "A"
        ).indices[0]
        == "l"
    )


def test_keeps_definitions_that_no_other_expression_uses() -> None:
    expression = Expression(Tensor("Y", ["i"]), Add([Tensor("A", ["i"]), Tensor("B", ["i"])]))

    result = substitute_expressions([expression], metadata())

    assert len(result) == 1
    assert result[0].lhs == expression.lhs
    assert result[0].rhs == expression.rhs


def test_rejects_dependency_cycles() -> None:
    first = Expression(Tensor("A", ["i"]), Tensor("B", ["i"]))
    second = Expression(Tensor("B", ["i"]), Tensor("A", ["i"]))

    with pytest.raises(ValueError, match="cycle"):
        substitute_expressions([first, second], metadata())
