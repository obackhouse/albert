import json
from io import StringIO

import pytest

from albert import (
    Add,
    EinsumEncoder,
    EinsumFunctionEncoder,
    Encoder,
    EncodingMethod,
    Expression,
    LatexEncoder,
    Metadata,
    Mul,
    Node,
    ReadableEncoder,
    Scalar,
    Tensor,
    TreeEncoder,
)


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


def test_node_encode_defaults_to_json_while_format_defaults_to_readable(capsys) -> None:
    node = Tensor("A", ["i"])

    assert node.format() == "A[i]"
    node.encode()

    encoded = capsys.readouterr().out
    assert encoded.endswith("\n")
    assert json.loads(encoded) == {"type": "tensor", "symbol": "A", "indices": ["i"]}


def test_format_and_encode_reject_non_encoder_arguments() -> None:
    node = Tensor("A", ["i"])

    with pytest.raises(TypeError, match="encoder must be"):
        node.format(12)


def test_format_and_encode_accept_encoding_method_strings_and_enums(capsys) -> None:
    node = Tensor("A", ["i"])
    expression = Expression(Tensor("out", ["i"]), node)

    assert node.format("readable") == "A[i]"
    assert node.format(EncodingMethod.LATEX) == r"A_{i}"
    assert expression.format("tree") == "Expression\n├── out[i]\n└── A[i]"

    node.encode("latex")
    assert capsys.readouterr().out == "A_{i}\n"


def test_encodes_tensor_contraction_in_readable_notation() -> None:
    x = Tensor("x", ["i", "j", "k"])
    y = Tensor("y", ["k", "j", "l"])
    expression = Expression(Tensor("z", ["i", "l"]), Mul([x, y]))
    buffer = StringIO()

    ReadableEncoder(buffer)(expression)

    assert buffer.getvalue() == "z[i, l] = x[i, j, k] y[k, j, l]\n"


def test_einsum_function_encoder_orders_assignments_and_deletes_dead_intermediates() -> None:
    import numpy as np

    metadata = Metadata(indices={"o": ["i", "j"], "v": ["k"]}, sizes={"o": 2, "v": 3})
    intermediate = Expression(
        Tensor("tmp", ["i", "k"]),
        Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j", "k"])]),
    )
    result = Expression(Tensor("result", ["i", "k"]), Tensor("tmp", ["i", "k"]))
    encoder = EinsumFunctionEncoder(StringIO(), metadata=metadata)

    assert encoder.input_keys([result, intermediate]) == ("A", "B")
    generated = encoder.format_function([result, intermediate])

    assert "def evaluate(**kwargs):" in generated
    assert "kwargs = SimpleNamespace(**kwargs)" in generated
    assert "A = kwargs.A" in generated
    assert "B = kwargs.B" in generated
    assert 'np.einsum("ij,jk->ik", A, B' in generated
    assert generated.index("tmp = np.einsum") < generated.index("result = tmp")
    assert "del tmp" in generated
    assert "return {'result': result}" in generated
    compile(generated, "<einsum function>", "exec")
    namespace: dict[str, object] = {}
    exec(generated, namespace)
    evaluate = namespace["evaluate"]
    assert callable(evaluate)
    arrays = {
        "A": np.arange(4, dtype=np.float64).reshape(2, 2),
        "B": np.arange(6, dtype=np.float64).reshape(2, 3),
    }
    results = evaluate(**arrays)
    np.testing.assert_allclose(results["result"], arrays["A"] @ arrays["B"])


def test_einsum_function_encoder_schedules_additive_terms_independently() -> None:
    import numpy as np

    metadata = Metadata(indices={"o": ["i"]}, sizes={"o": 3})
    result = Expression(
        Tensor("result", ["i"]),
        Add([Tensor("left", ["i"]), Mul([Scalar(-2), Tensor("right", ["i"])])]),
    )
    final = Expression(Tensor("final", ["i"]), Mul([Tensor("result", ["i"]), Tensor("E", [])]))
    left = Expression(Tensor("left", ["i"]), Tensor("A", ["i"]))
    right = Expression(Tensor("right", ["i"]), Tensor("B", ["i"]))
    later = Expression(
        Tensor("later", ["i"]),
        Mul([Tensor("C", ["i"]), Tensor("D", [])]),
    )
    generated = EinsumFunctionEncoder(StringIO(), metadata=metadata).format_function(
        [final, result, left, right, later], outputs=("final",)
    )

    assert generated.index("left = A") < generated.index("result = np.copy(left)")
    assert generated.index("result = np.copy(left)") < generated.index("del left")
    assert generated.index("del left") < generated.index("right = B")
    assert generated.index("result -= 2.0 * right") < generated.index("del right")
    assert generated.index("result -= 2.0 * right") < generated.index('final = np.einsum("i,->i"')
    assert generated.index('final = np.einsum("i,->i"') < generated.index("del result")
    assert generated.index("del result") < generated.index("later = np.einsum")

    namespace: dict[str, object] = {}
    exec(generated, namespace)
    arrays = {
        "A": np.array([1.0, 2.0, 3.0]),
        "B": np.array([4.0, 5.0, 6.0]),
        "C": np.array([2.0, 3.0, 4.0]),
        "D": np.array(2.0),
        "E": np.array(3.0),
    }
    np.testing.assert_array_equal(
        namespace["evaluate"](**arrays)["final"], (arrays["A"] - 2 * arrays["B"]) * arrays["E"]
    )


def test_einsum_function_encoder_can_contract_ready_additive_terms_first() -> None:
    metadata = Metadata(indices={"o": ["i"]}, sizes={"o": 2})
    result = Expression(Tensor("result", ["i"]), Add([Tensor("left", ["i"]), Tensor("B", ["i"])]))
    left = Expression(Tensor("left", ["i"]), Tensor("A", ["i"]))

    generated = EinsumFunctionEncoder(StringIO(), metadata=metadata).format_function(
        [result, left], outputs=("result",)
    )

    assert generated.index("result = np.copy(B)") < generated.index("left = A")
    assert generated.index("left = A") < generated.index("result += left")


def test_einsum_function_encoder_schedules_ready_consumers_before_independent_tasks() -> None:
    metadata = Metadata(indices={"o": ["i"]}, sizes={"o": 2})
    source = Expression(Tensor("source", ["i"]), Tensor("A", ["i"]))
    independent = Expression(Tensor("independent", ["i"]), Tensor("B", ["i"]))
    middle = Expression(Tensor("middle", ["i"]), Tensor("source", ["i"]))
    consumer = Expression(Tensor("consumer", ["i"]), Tensor("middle", ["i"]))

    generated = EinsumFunctionEncoder(StringIO(), metadata=metadata).format_function(
        [source, independent, middle, consumer]
    )

    source_position = generated.index("source = A")
    middle_position = generated.index("middle = source")
    consumer_position = generated.index("consumer = middle")
    independent_position = generated.index("independent = B")
    assert source_position < middle_position < consumer_position < independent_position


def test_einsum_function_encoder_supports_space_specific_input_formatting() -> None:
    from types import SimpleNamespace

    import numpy as np

    metadata = Metadata(indices={"o": ["i"], "v": ["a"]}, sizes={"o": 2, "v": 3})
    expression = Expression(
        Tensor("result", ["i", "a"]),
        Mul([Tensor("A", ["i"]), Tensor("A", ["a"])]),
    )
    encoder = EinsumFunctionEncoder(
        StringIO(),
        metadata=metadata,
        einsum_encoder=EinsumEncoder(tensor_formatter=lambda symbol, spaces: f"{symbol}.{''.join(spaces)}"),
    )
    generated = encoder.format_function([expression])

    assert "def evaluate(**kwargs):" in generated
    assert encoder.input_keys([expression]) == ("A",)
    assert "A = kwargs.A" in generated
    assert "A.o, A.v" in generated
    namespace: dict[str, object] = {}
    exec(generated, namespace)
    evaluate = namespace["evaluate"]
    assert callable(evaluate)
    arrays = {"o": np.array([1.0, 2.0]), "v": np.array([3.0, 4.0, 5.0])}
    result = evaluate(A=SimpleNamespace(**arrays))
    np.testing.assert_allclose(result["result.ov"], np.outer(arrays["o"], arrays["v"]))


def test_einsum_function_encoder_can_return_multiple_named_outputs() -> None:
    metadata = Metadata(indices={"o": ["i"]}, sizes={"o": 2})
    first = Expression(Tensor("first", ["i"]), Tensor("A", ["i"]))
    second = Expression(Tensor("second", ["i"]), Tensor("B", ["i"]))
    generated = EinsumFunctionEncoder(StringIO(), metadata=metadata).format_function([first, second])

    assert "def evaluate(**kwargs):" in generated
    assert "return {'first': first, 'second': second}" in generated


def test_einsum_function_encoder_uses_transpose_function() -> None:
    metadata = Metadata(indices={"o": ["i", "j"]}, sizes={"o": 2})
    expression = Expression(Tensor("result", ["j", "i"]), Tensor("A", ["i", "j"]))
    encoder = EinsumFunctionEncoder(StringIO(), metadata=metadata)

    assert "result = np.transpose(A, (1, 0))" in encoder.format_function([expression])


def test_einsum_function_uses_formatted_names_for_same_symbol_in_different_spaces() -> None:
    import numpy as np

    metadata = Metadata(indices={"o": ["i"], "v": ["a"]}, sizes={"o": 2, "v": 3})
    expressions = [
        Expression(Tensor("A", ["i"]), Tensor("X", ["i"])),
        Expression(Tensor("A", ["a"]), Tensor("Y", ["a"])),
    ]
    encoder = EinsumFunctionEncoder(
        StringIO(),
        metadata=metadata,
        einsum_encoder=EinsumEncoder(
            tensor_formatter=lambda symbol, spaces: f"{symbol}_{''.join(spaces)}" if symbol == "A" else symbol
        ),
    )
    generated = encoder.format_function(expressions)

    assert "X = kwargs.X" in generated
    assert "Y = kwargs.Y" in generated
    assert "A_o = X" in generated
    assert "A_v = Y" in generated
    assert "return {'A_o': A_o, 'A_v': A_v}" in generated
    namespace: dict[str, object] = {}
    exec(generated, namespace)
    evaluate = namespace["evaluate"]
    assert callable(evaluate)
    arrays = {"X": np.array([1.0, 2.0]), "Y": np.array([3.0, 4.0, 5.0])}

    results = evaluate(**arrays)

    np.testing.assert_array_equal(results["A_o"], arrays["X"])
    np.testing.assert_array_equal(results["A_v"], arrays["Y"])


def test_einsum_function_encoder_customizes_imports() -> None:
    metadata = Metadata(indices={"o": ["i"]}, sizes={"o": 2})
    expression = Expression(Tensor("result", ["i"]), Tensor("A", ["i"]))
    encoder = EinsumFunctionEncoder(
        StringIO(),
        metadata=metadata,
        imports=("import jax.numpy as jnp",),
        einsum_encoder=EinsumEncoder(einsum_function="jnp.einsum", copy_function="jnp.copy"),
    )

    generated = encoder.format_function([expression])

    assert generated.startswith("import jax.numpy as jnp\nfrom types import SimpleNamespace\n\ndef evaluate(**kwargs):")
    assert "import numpy as np" not in generated
    assert "A = kwargs.A" in generated
    assert "result = A" in generated


def test_einsum_function_encoder_can_omit_imports_per_function() -> None:
    metadata = Metadata(indices={"o": ["i"]}, sizes={"o": 2})
    expression = Expression(Tensor("result", ["i"]), Tensor("A", ["i"]))
    buffer = StringIO()
    encoder = EinsumFunctionEncoder(buffer, metadata=metadata)

    formatted = encoder.format_function([expression], with_imports=False)
    encoder.encode_function([expression], function_name="again", with_imports=False)

    assert formatted.startswith("def evaluate(**kwargs):")
    assert "import numpy as np" not in formatted
    assert buffer.getvalue().startswith("def again(**kwargs):")
    assert "import numpy as np" not in buffer.getvalue()


def test_einsum_function_outputs_can_feed_an_iterative_function() -> None:
    import numpy as np

    metadata = Metadata(
        indices={"i": ["i"], "j": ["j"], "k": ["k"], "l": ["l"]},
        sizes={"i": 2, "j": 2, "k": 3, "l": 4},
    )
    static_expression = Expression(
        Tensor("ab", ["i", "k"]),
        Mul([Tensor("A", ["i", "j"]), Tensor("B", ["j", "k"])]),
    )
    iterative_expression = Expression(
        Tensor("result", ["i", "l"]),
        Mul([Tensor("ab", ["i", "k"]), Tensor("C", ["k", "l"])]),
    )
    static_code = EinsumFunctionEncoder(StringIO(), metadata=metadata, function_name="static").format_function(
        [static_expression]
    )
    iterative_code = EinsumFunctionEncoder(StringIO(), metadata=metadata, function_name="iterate").format_function(
        [iterative_expression]
    )
    namespace: dict[str, object] = {}
    exec(static_code + iterative_code, namespace)
    static = namespace["static"]
    iterate = namespace["iterate"]
    assert callable(static)
    assert callable(iterate)
    arrays = {
        "A": np.arange(4, dtype=np.float64).reshape(2, 2),
        "B": np.arange(6, dtype=np.float64).reshape(2, 3),
        "C": np.arange(12, dtype=np.float64).reshape(3, 4),
    }

    static_arrays = static(**arrays)
    result = iterate(**{**static_arrays, "C": arrays["C"]})

    np.testing.assert_allclose(result["result"], arrays["A"] @ arrays["B"] @ arrays["C"])


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


@pytest.mark.parametrize(
    ("rhs", "lhs", "expected"),
    [
        (
            Mul([Tensor("y", ["i", "j", "k"]), Tensor("z", ["j", "k", "l"])]),
            Tensor("x", ["i", "l"]),
            r"x_{il} = \sum_{jk} y_{ijk} z_{jkl}",
        ),
        (
            Tensor("a", ["i", "i"]),
            Tensor("out", []),
            r"out = \sum_{i} a_{ii}",
        ),
        (
            Mul([Add([Tensor("x", ["i"]), Tensor("y", ["i"])]), Tensor("z", ["i"])]),
            Tensor("out", []),
            r"out = \sum_{i} \left(x_{i} + y_{i}\right) z_{i}",
        ),
        (
            Mul(
                [
                    Mul([Tensor("a", ["i", "j"]), Tensor("b", ["j", "k"])]),
                    Tensor("d", ["k"]),
                ]
            ),
            Tensor("out", ["i"]),
            r"out_{i} = \sum_{k} \left(\sum_{j} a_{ij} b_{jk}\right) d_{k}",
        ),
        (
            Add([Tensor("x", ["i"]), Mul([Scalar(-1), Tensor("y", ["i"])])]),
            Tensor("out", ["i"]),
            r"out_{i} = x_{i} - y_{i}",
        ),
    ],
)
def test_latex_encoder_formats_nodes_and_expressions(rhs, lhs, expected: str) -> None:
    expression = Expression(lhs, rhs)
    buffer = StringIO()

    LatexEncoder(buffer)(expression)

    assert buffer.getvalue() == expected + "\n"
    assert expression.format(LatexEncoder) == expected


def test_latex_encoder_supports_custom_formatting_options() -> None:
    expression = Expression(
        Tensor("x", ["i", "l"]),
        Mul([Tensor("y", ["i", "j", "k"]), Tensor("z", ["j", "k", "l"])]),
    )

    assert (
        expression.format(
            LatexEncoder(
                index_separator=", ",
                multiplication_sign=r"\cdot",
                tensor_style=r"\mathbf{{{}}}",
            )
        )
        == r"\mathbf{x}_{i, l} = \sum_{j, k} \mathbf{y}_{i, j, k} \cdot \mathbf{z}_{j, k, l}"
    )


def test_latex_encoder_formats_scalar_products_and_standalone_nodes() -> None:
    product = Mul([Scalar(2), Scalar(3)])
    negative_tensor = Mul([Scalar(-1), Tensor("x", ["i"])])

    assert product.format(LatexEncoder) == "2 3"
    assert negative_tensor.format(LatexEncoder) == r"-x_{i}"


@pytest.mark.parametrize(
    ("rhs", "expected"),
    [
        (Mul([Scalar(2), Tensor("x", ["i"])]), r"out_{i} = 2 x_{i}"),
        (Mul([Tensor("x", ["i"]), Scalar(2)]), r"out_{i} = 2 x_{i}"),
        (Mul([Scalar(-2), Tensor("x", ["i"])]), r"out_{i} = -2 x_{i}"),
        (Mul([Scalar(2.5), Tensor("x", ["i"])]), r"out_{i} = 2.5 x_{i}"),
        (
            Add([Tensor("x", ["i"]), Mul([Scalar(-2), Tensor("y", ["i"])])]),
            r"out_{i} = x_{i} - 2 y_{i}",
        ),
    ],
)
def test_latex_encoder_formats_scalar_coefficients_before_tensor_factors(rhs, expected: str) -> None:
    expression = Expression(Tensor("out", ["i"]), rhs)

    assert expression.format(LatexEncoder) == expected


def test_tree_encoder_formats_an_expression_as_a_connector_tree() -> None:
    expression = Expression(Tensor("out", []), Mul([Tensor("x", ["i"]), Tensor("y", ["i"])]))

    assert expression.format(TreeEncoder) == "Expression\n├── out\n└── Mul\n    ├── x[i]\n    └── y[i]"


def test_tree_encoder_formats_internal_nodes_and_scalar_leaves() -> None:
    x = Tensor("x", ["i"])
    expression = Expression(Tensor("out", ["i"]), Add([Mul([Scalar(2), x]), x]))

    assert expression.format(TreeEncoder) == (
        "Expression\n├── out[i]\n└── Add\n    ├── Mul\n    │   ├── 2.0\n    │   └── x[i]\n    └── x[i]"
    )


def test_tree_encoder_supports_custom_connectors_and_minimum_indent() -> None:
    expression = Expression(
        Tensor("out", []),
        Add([Tensor("x", []), Tensor("y", [])]),
    )
    encoder = TreeEncoder(connectors=("-", "|", "+", "\\"), indent=1)

    assert expression.format(encoder) == "Expression\n+ out\n\\ Add\n  + x\n  \\ y"


def test_einsum_encoder_formats_sums_as_assignments() -> None:
    y, z = Tensor("y", ["i", "k"]), Tensor("z", ["k", "l"])
    a, b = Tensor("a", ["i", "k"]), Tensor("b", ["k", "l"])
    expression = Expression(Tensor("x", ["i", "l"]), Add([Mul([y, z]), Mul([a, b])]))
    buffer = StringIO()

    EinsumEncoder(buffer)(expression)

    assert buffer.getvalue() == (
        'x = np.einsum("ik,kl->il", y, z, optimize=True)\nx += np.einsum("ik,kl->il", a, b, optimize=True)\n'
    )


def test_einsum_encoder_can_lower_three_index_copy_deltas() -> None:
    metadata = Metadata(indices={"o": ["i", "j", "k"]}, sizes={"o": 3})
    expression = Expression(
        Tensor("out", ["i"]),
        Mul([Tensor("A", ["j", "k"]), Tensor("δ", ["i", "j", "k"])]),
    )

    assert expression.format(EinsumEncoder(metadata=metadata)) == ('out = np.einsum("jk,ijk->i", A, δ, optimize=True)')
    assert expression.format(EinsumEncoder(metadata=metadata, lower_copy_deltas=True)) == (
        'out = np.einsum("ii->i", A, optimize=True)'
    )


def test_einsum_encoder_lowering_preserves_equal_output_axes() -> None:
    import numpy as np

    metadata = Metadata(indices={"o": ["i", "j", "k"]}, sizes={"o": 3})
    expression = Expression(
        Tensor("out", ["i", "j"]),
        Mul([Tensor("A", ["k"]), Tensor("δ", ["i", "j", "k"])]),
    )
    source = EinsumFunctionEncoder(
        StringIO(),
        metadata=metadata,
        einsum_encoder=EinsumEncoder(metadata=metadata, lower_copy_deltas=True),
    ).format_function([expression])

    assert 'np.einsum("i,ij->ij", A, np.eye(3, dtype=bool)' in source
    assert "δ" not in source
    namespace: dict[str, object] = {}
    exec(source, namespace)
    array = np.arange(3, dtype=np.float32)
    result = namespace["evaluate"](A=array)["out"]
    np.testing.assert_array_equal(result, np.diag(array))


def test_einsum_encoder_lowering_uses_a_ones_vector_when_output_axis_would_vanish() -> None:
    metadata = Metadata(indices={"o": ["i", "j"]}, sizes={"o": 2})
    expression = Expression(Tensor("out", ["i"]), Tensor("δ", ["i", "j", "j"]))

    assert expression.format(EinsumEncoder(metadata=metadata, lower_copy_deltas=True)) == (
        "out = np.ones(2, dtype=bool)"
    )


def test_einsum_encoder_can_use_matmul_for_rank_two_contractions() -> None:
    a, b = Tensor("a", ["i", "k"]), Tensor("b", ["k", "j"])
    expression = Expression(Tensor("out", ["i", "j"]), Mul([a, b]))

    assert expression.format(EinsumEncoder(use_matmul=True)) == "out = a @ b"


@pytest.mark.parametrize(
    ("a_indices", "b_indices", "out_indices", "expected"),
    [
        (("k", "i"), ("k", "j"), ("i", "j"), "out = a.T @ b"),
        (("i", "k"), ("j", "k"), ("i", "j"), "out = a @ b.T"),
        (("k", "i"), ("j", "k"), ("j", "i"), "out = (a.T @ b.T).T"),
        (("i", "k"), ("k", "j"), ("j", "i"), "out = (a @ b).T"),
    ],
)
def test_einsum_encoder_transposes_rank_two_matmul_operands(
    a_indices: tuple[str, str],
    b_indices: tuple[str, str],
    out_indices: tuple[str, str],
    expected: str,
) -> None:
    expression = Expression(
        Tensor("out", out_indices),
        Mul([Tensor("a", a_indices), Tensor("b", b_indices)]),
    )

    assert expression.format(EinsumEncoder(use_matmul=True)) == expected


def test_einsum_encoder_keeps_einsum_for_batched_contractions() -> None:
    a = Tensor("a", ["batch", "i", "k"])
    b = Tensor("b", ["batch", "k", "j"])
    expression = Expression(Tensor("out", ["i", "j"]), Mul([a, b]))

    assert expression.format(EinsumEncoder(use_matmul=True)) == ('out = np.einsum("aik,akj->ij", a, b, optimize=True)')


def test_einsum_function_encoder_propagates_matmul_option() -> None:
    import numpy as np

    metadata = Metadata(indices={"o": ["i", "k", "j"]}, sizes={"o": 2})
    expression = Expression(
        Tensor("out", ["i", "j"]),
        Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k", "j"])]),
    )
    source = EinsumFunctionEncoder(
        StringIO(),
        metadata=metadata,
        einsum_encoder=EinsumEncoder(use_matmul=True),
    ).format_function([expression])

    assert "out = A @ B" in source
    namespace: dict[str, object] = {}
    exec(source, namespace)
    a = np.arange(4, dtype=np.float64).reshape(2, 2)
    b = np.arange(4, dtype=np.float64).reshape(2, 2)
    result = namespace["evaluate"](A=a, B=b)
    np.testing.assert_array_equal(result["out"], a @ b)


def test_einsum_function_encoder_evaluates_transposed_matmul() -> None:
    import numpy as np

    metadata = Metadata(
        indices={"o": ["i", "j"], "x": ["k"]},
        sizes={"o": 2, "x": 3},
    )
    expression = Expression(
        Tensor("out", ["j", "i"]),
        Mul([Tensor("A", ["k", "i"]), Tensor("B", ["j", "k"])]),
    )
    source = EinsumFunctionEncoder(
        StringIO(),
        metadata=metadata,
        einsum_encoder=EinsumEncoder(use_matmul=True),
    ).format_function([expression])

    assert "out = (A.T @ B.T).T" in source
    namespace: dict[str, object] = {}
    exec(source, namespace)
    a = np.arange(6, dtype=np.float64).reshape(3, 2)
    b = np.arange(6, dtype=np.float64).reshape(2, 3)
    result = namespace["evaluate"](A=a, B=b)
    np.testing.assert_array_equal(result["out"], np.einsum("ki,jk->ji", a, b))


def test_einsum_encoder_copies_a_view_before_accumulating() -> None:
    z = Tensor("z", ["i", "j"])
    a, b = Tensor("a", ["i", "k"]), Tensor("b", ["j", "k"])
    expression = Expression(Tensor("x", ["j", "i"]), Add([z, Mul([a, b])]))

    assert expression.format(EinsumEncoder) == (
        'x = np.copy(np.transpose(z, (1, 0)))\nx += np.einsum("ik,jk->ji", a, b, optimize=True)'
    )


def test_einsum_encoder_falls_back_to_einsum_when_transpose_function_is_none() -> None:
    z = Tensor("z", ["i", "j"])
    expression = Expression(Tensor("out", ["j", "i"]), z)

    assert expression.format(EinsumEncoder(transpose_function=None)) == ('out = np.einsum("ij->ji", z, optimize=True)')


def test_einsum_encoder_uses_transpose_function_for_accumulated_term() -> None:
    a, b = Tensor("a", ["i"]), Tensor("b", ["j"])
    z = Tensor("z", ["i", "j"])
    expression = Expression(Tensor("out", ["j", "i"]), Add([Mul([a, b]), z]))

    assert expression.format(EinsumEncoder()) == (
        'out = np.einsum("i,j->ji", a, b, optimize=True)\nout += np.transpose(z, (1, 0))'
    )


def test_einsum_encoder_subtracts_negative_accumulated_terms() -> None:
    a, b = Tensor("a", ["i"]), Tensor("b", ["i"])
    expression = Expression(Tensor("out", ["i"]), Add([a, Mul([Scalar(-2), b])]))

    assert expression.format(EinsumEncoder()) == "out = np.copy(a)\nout -= 2.0 * b"


def test_einsum_encoder_uses_tensor_directly_for_identity_indices() -> None:
    tensor = Tensor("a", ["i", "j"])
    expression = Expression(Tensor("out", ["i", "j"]), tensor)

    assert tensor.format(EinsumEncoder) == "a"
    assert expression.format(EinsumEncoder) == "out = a"


def test_einsum_encoder_only_copies_before_in_place_accumulation() -> None:
    a, b = Tensor("a", ["i"]), Tensor("b", ["i"])
    single_term = Expression(Tensor("single", ["i"]), a)
    expression = Expression(Tensor("out", ["i"]), Add([a, b]))

    assert single_term.format(EinsumEncoder()) == "single = a"
    assert expression.format(EinsumEncoder) == "out = np.copy(a)\nout += b"


def test_einsum_encoder_supports_custom_options_and_method_name() -> None:
    z = Tensor("z", ["i", "j"])
    a, b = Tensor("a", ["i", "k"]), Tensor("b", ["j", "k"])
    expression = Expression(Tensor("x", ["j", "i"]), Add([z, Mul([a, b])]))
    encoder = EinsumEncoder(
        einsum_function="einsum",
        einsum_kwargs={"optimize": False},
        copy_function="copy",
    )

    assert expression.format(encoder) == (
        'x = copy(np.transpose(z, (1, 0)))\nx += einsum("ik,jk->ji", a, b, optimize=False)'
    )
    assert expression.format("einsum") == expression.format(EinsumEncoder)


def test_einsum_encoder_distributes_products_over_sums_and_formats_scalars() -> None:
    a, b = Tensor("a", ["i"]), Tensor("b", ["i"])
    z = Tensor("z", ["j"])
    expression = Expression(
        Tensor("out", ["i", "j"]),
        Mul([Add([a, b]), Scalar(-2), z]),
    )

    assert expression.format(EinsumEncoder(einsum_kwargs={})) == (
        'out = -2.0 * np.einsum("i,j->ij", a, z)\nout -= 2.0 * np.einsum("i,j->ij", b, z)'
    )


def test_einsum_encoder_allows_space_specific_tensor_formatting() -> None:
    metadata = Metadata(
        indices={"o": ["i", "j"], "v": ["k"]},
        sizes={"o": 2, "v": 3},
    )
    expression = Expression(
        Tensor("out", ["i", "j"]),
        Mul([Tensor("eri", ["i", "k"]), Tensor("eri", ["k", "j"])]),
    )

    assert expression.format(
        EinsumEncoder(
            metadata=metadata,
            tensor_formatter=lambda symbol, spaces: f"{symbol}.{''.join(spaces)}",
        )
    ) == ('out.oo = np.einsum("ik,kj->ij", eri.ov, eri.vo, optimize=True)')
