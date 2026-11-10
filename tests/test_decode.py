import json
from io import StringIO

import pytest

from albert import Add, DecodingMethod, Expression, JsonDecoder, JsonEncoder, Mul, Node, Scalar, Tensor


def test_json_node_encoder_and_node_decode_round_trip() -> None:
    expected = Add(
        [
            Mul([Scalar(2.0), Tensor("A", ["i", "j"])]),
            Tensor("B", ["i", "j"]),
        ]
    )

    encoded = expected.format(JsonEncoder)
    decoded = Node.decode(encoded)

    assert json.loads(encoded) == {
        "type": "add",
        "children": [
            {
                "type": "mul",
                "children": [
                    {"type": "scalar", "value": 2.0},
                    {"type": "tensor", "symbol": "A", "indices": ["i", "j"]},
                ],
            },
            {"type": "tensor", "symbol": "B", "indices": ["i", "j"]},
        ],
    }
    assert decoded == expected


def test_node_decode_accepts_a_decoder_instance_or_subclass() -> None:
    encoded = Tensor("A", ["i"]).format(JsonEncoder)

    assert Node.decode(encoded, JsonDecoder) == Tensor("A", ["i"])
    assert Node.decode(encoded, JsonDecoder()) == Tensor("A", ["i"])


def test_decode_accepts_the_json_method_string_or_enum() -> None:
    encoded = Tensor("A", ["i"]).format(JsonEncoder)

    assert Node.decode(encoded, "json") == Tensor("A", ["i"])
    assert Node.decode(encoded, DecodingMethod.JSON) == Tensor("A", ["i"])


def test_node_decode_is_a_classmethod_and_respects_subclasses() -> None:
    encoded = Tensor("A", ["i"]).format(JsonEncoder)

    assert Node.decode.__self__ is Node
    assert Tensor.decode(encoded) == Tensor("A", ["i"])
    with pytest.raises(TypeError, match="incompatible with the requested class"):
        Scalar.decode(encoded)


def test_json_expression_encoder_and_decoder_round_trip() -> None:
    expected = Expression(
        Tensor("Y", ["i"]),
        Add([Tensor("A", ["i"]), Mul([Scalar(-2.5), Tensor("B", ["i"])])]),
    )
    encoded = expected.format(JsonEncoder)
    decoded = JsonDecoder()(encoded)

    assert isinstance(decoded, Expression)
    assert decoded.lhs == expected.lhs
    assert decoded.rhs == expected.rhs


def test_expression_decode_is_a_classmethod_with_json_as_default() -> None:
    expected = Expression(Tensor("Y", ["i"]), Tensor("A", ["i"]))
    encoded = expected.format(JsonEncoder)

    decoded = Expression.decode(encoded)

    assert Expression.decode.__self__ is Expression
    assert isinstance(decoded, Expression)
    assert decoded.lhs == expected.lhs
    assert decoded.rhs == expected.rhs
    decoded_by_method = Expression.decode(encoded, "json")
    assert decoded_by_method.lhs == expected.lhs
    assert decoded_by_method.rhs == expected.rhs


def test_json_encoder_supports_compact_formatting_and_buffer_write() -> None:
    node = Tensor("A", ["i"])
    compact_encoder = JsonEncoder(StringIO(), indent=None)
    compact = node.format(compact_encoder)
    assert "\n" not in compact
    output = StringIO()
    node.encode(JsonEncoder(output, indent=None))
    assert output.getvalue() == compact + "\n"


def test_node_decode_rejects_invalid_json_and_unknown_node_types() -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        Node.decode("{")
    with pytest.raises(ValueError, match="unsupported node type"):
        Node.decode('{"type":"future"}')


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"type":"scalar","value":true}', "finite number"),
        ('{"type":"scalar","value":NaN}', "finite number"),
        ('{"type":"tensor","symbol":3,"indices":[]}', "tensor symbol"),
        ('{"type":"tensor","symbol":"A","indices":[1]}', "tensor indices"),
        ('{"type":"add","children":null}', "children must be a list"),
    ],
)
def test_node_decode_validates_node_payloads(payload: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Node.decode(payload)


def test_node_decode_rejects_non_decoder_arguments() -> None:
    with pytest.raises(TypeError, match="decoder must be"):
        Node.decode('{"type":"scalar","value":1}', 12)  # type: ignore[arg-type]


def test_decode_rejects_unsupported_method_names() -> None:
    with pytest.raises(ValueError, match="DecodingMethod"):
        Node.decode('{"type":"scalar","value":1}', "latex")
