from io import StringIO

import pytest

from albert import Add, Encoder, Expression, Mul, Node, ReadableEncoder, Scalar, Tensor


class RecordingEncoder(Encoder):
    def format_node(self, node: Node) -> str:
        return f"node:{type(node).__name__}"

    def format_expression(self, expression: Expression) -> str:
        return f"expression:{expression.lhs.symbol}"


def test_encoder_dispatches_nodes_and_expressions_to_the_buffer() -> None:
    buffer = StringIO()
    encoder = RecordingEncoder(buffer)
    expression = Expression(Tensor("Y", []), Scalar(2))

    encoder(Scalar(2))
    encoder(expression)

    assert buffer.getvalue() == "node:Scalar\nexpression:Y\n"


def test_node_and_expression_repr_use_readable_format_without_writing(capsys) -> None:
    node = Add([Tensor("A", ["i"]), Mul([Scalar(-2), Tensor("B", ["i"])])])
    expression = Expression(Tensor("Y", ["i"]), node)

    assert repr(node) == "A(i) - 2 B(i)"
    assert repr(expression) == "Y(i) = A(i) - 2 B(i)"
    assert capsys.readouterr().out == ""


def test_repr_uses_compact_index_separators_and_parenthesizes_sums() -> None:
    a = Tensor("A", ["i", "k"])
    b = Tensor("B", ["i", "k"])
    c = Tensor("C", ["k", "j"])
    node = Mul([Add([a, b]), c])
    expression = Expression(Tensor("Y", ["i", "j"]), node)

    assert repr(node) == "(A(i,k) + B(i,k)) C(k,j)"
    assert repr(expression) == "Y(i,j) = (A(i,k) + B(i,k)) C(k,j)"


def test_format_returns_strings_with_default_class_or_encoder_instance() -> None:
    node = Tensor("A", ["i", "j"])
    expression = Expression(Tensor("Y", ["i"]), Tensor("A", ["i"]))
    custom_encoder = ReadableEncoder(StringIO(), index_separator=" | ")

    assert node.format() == "A[i, j]"
    assert node.format(ReadableEncoder) == "A[i, j]"
    assert node.format(custom_encoder) == "A[i | j]"
    assert expression.format() == "Y[i] = A[i]"


def test_encode_writes_through_the_supplied_encoder_instance() -> None:
    node_buffer = StringIO()
    expression_buffer = StringIO()
    node = Tensor("A", ["i"])
    expression = Expression(Tensor("Y", ["i"]), node)

    assert node.encode(ReadableEncoder(node_buffer)) is None
    expression.encode(ReadableEncoder(expression_buffer))

    assert node_buffer.getvalue() == "A[i]\n"
    assert expression_buffer.getvalue() == "Y[i] = A[i]\n"


def test_format_and_encode_reject_non_encoder_arguments() -> None:
    node = Tensor("A", ["i"])

    with pytest.raises(TypeError, match="encoder must be"):
        node.format("readable")


def test_encodes_tensor_contraction_in_readable_notation() -> None:
    x = Tensor("x", ["i", "j", "k"])
    y = Tensor("y", ["k", "j", "l"])
    expression = Expression(Tensor("z", ["i", "l"]), Mul([x, y]))
    buffer = StringIO()

    ReadableEncoder(buffer)(expression)

    assert buffer.getvalue() == "z[i, l] = x[i, j, k] y[k, j, l]\n"


def test_formats_addition_subtraction_and_parentheses() -> None:
    x, y = (Tensor(symbol, ["i", "k"]) for symbol in ("x", "y"))
    z = Tensor("z", ["k", "j"])
    w = Tensor("w", ["i", "j"])
    expression = Expression(Tensor("out", ["i", "j"]), Add([Mul([Add([x, y]), z]), Mul([Scalar(-1), w])]))
    buffer = StringIO()

    ReadableEncoder(buffer)(expression)

    assert buffer.getvalue() == "out[i, j] = (x[i, k] + y[i, k]) z[k, j] - w[i, j]\n"


def test_encodes_scalar_products_and_zero_rank_tensors() -> None:
    scalar_expression = Expression(Tensor("out", []), Mul([Scalar(2), Scalar(3)]))
    tensor_expression = Expression(Tensor("out", []), Tensor("unit", []))
    buffer = StringIO()
    encoder = ReadableEncoder(buffer)

    encoder(scalar_expression)
    encoder(tensor_expression)

    assert buffer.getvalue() == "out = 2 * 3\nout = unit\n"


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (Scalar(0.0), "0"),
        (Scalar(-0.0), "-0"),
        (Scalar(3.125), "3.125"),
        (Tensor("unit", []), "unit"),
    ],
)
def test_encodes_scalar_and_zero_rank_tensor_leaves(node, expected: str) -> None:
    buffer = StringIO()

    ReadableEncoder(buffer)(node)

    assert buffer.getvalue() == f"{expected}\n"


def test_parenthesizes_additions_and_uses_custom_tensor_index_delimiters() -> None:
    x, y = (Tensor(symbol, ["i", "j"]) for symbol in ("x", "y"))
    z, w = (Tensor(symbol, ["k", "l"]) for symbol in ("z", "w"))
    expression = Expression(
        Tensor("out", ["i", "j", "k", "l"]),
        Mul([Add([x, y]), Add([z, w])]),
    )
    buffer = StringIO()

    ReadableEncoder(buffer, bracket_style=("{", "}"), index_separator=" | ")(expression)

    assert buffer.getvalue() == "out{i | j | k | l} = (x{i | j} + y{i | j}) (z{k | l} + w{k | l})\n"


@pytest.mark.parametrize(
    ("rhs", "expected"),
    [
        (lambda x, y: Mul([Scalar(-1), x]), "-x[i]"),
        (lambda x, y: Add([x, Mul([Scalar(-1), y])]), "x[i] - y[i]"),
        (lambda x, y: Add([Mul([Scalar(-1), x]), Mul([Scalar(-1), y])]), "-x[i] - y[i]"),
        (lambda x, y: Mul([Scalar(-2), x]), "-2 x[i]"),
        (lambda x, y: Mul([Mul([Scalar(-1), x]), Mul([Scalar(-1), y])]), "x[i] y[i]"),
    ],
)
def test_formats_signs(rhs, expected: str) -> None:
    x, y = Tensor("x", ["i"]), Tensor("y", ["i"])
    rhs_node = rhs(x, y)
    expression = Expression(Tensor("out", rhs_node.external_indices), rhs_node)
    buffer = StringIO()
    encoder = ReadableEncoder(buffer)

    encoder(expression)

    assert buffer.getvalue() == f"{encoder.format_node(expression.lhs)} = {expected}\n"


def test_flattens_nested_additions_without_deduplicating_terms() -> None:
    x, y, z = (Tensor(symbol, ["i"]) for symbol in ("x", "y", "z"))
    shared = Add([x, y])
    expression = Expression(Tensor("out", ["i"]), Add([shared, shared, z]))
    buffer = StringIO()

    ReadableEncoder(buffer)(expression)

    assert buffer.getvalue() == "out[i] = x[i] + y[i] + x[i] + y[i] + z[i]\n"
