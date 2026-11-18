import pytest

from albert._core import Add, Expression, Mul, Node, Scalar, Tensor
from albert.chemistry import PdaggerqDecoder


def _formatted_terms(node: Node) -> set[str]:
    children = node.children if isinstance(node, Add) else [node]
    return {child.format() for child in children}


def test_decodes_pdaggerq_terms_without_importing_pdaggerq() -> None:
    decoder = PdaggerqDecoder()
    encoded_terms = '[["+0.25", "t2(a,b,i,j)", "<i,j||a,b>"], ["-1.0", "t1(a,i)", "f(i,a)"]]'

    decoded = decoder.decode_node(encoded_terms)

    assert decoded == Add(
        [
            Mul([Scalar(0.25), Tensor("t2", ["i", "j", "a", "b"]), Tensor("v_a", ["i", "j", "a", "b"])]),
            Mul([Scalar(-1.0), Tensor("t1", ["i", "a"]), Tensor("f", ["i", "a"])]),
        ]
    )


def test_maps_antisymmetrized_and_ordinary_physicist_eris() -> None:
    decoder = PdaggerqDecoder()

    antisymmetrized = decoder.decode_node('[ ["<i,j||k,l>"] ]')
    ordinary = decoder.decode_node('[ ["<i,j|k,l>"] ]')

    assert antisymmetrized == Tensor("v_a", ["i", "j", "k", "l"])
    assert ordinary == Tensor("v", ["i", "j", "k", "l"])


def test_expands_pdaggerq_permutation_operators() -> None:
    decoder = PdaggerqDecoder()

    decoded = decoder.decode_node('[["+1.0", "P(i,j)", "f(i,a)", "t1(a,j)"]]')
    expected = Add(
        [
            Mul([Tensor("f", ["i", "a"]), Tensor("t1", ["j", "a"])]),
            Mul([Scalar(-1.0), Tensor("f", ["j", "a"]), Tensor("t1", ["i", "a"])]),
        ]
    )

    assert _formatted_terms(decoded) == _formatted_terms(expected)


def test_decodes_expression_envelope() -> None:
    decoder = PdaggerqDecoder()
    encoded = "{'lhs': 't1(a,i)', 'terms': [['+1.0', 'f(i,a)']]}"

    decoded = decoder(encoded)

    assert isinstance(decoded, Expression)
    assert decoded.lhs == Tensor("t1", ["i", "a"])
    assert decoded.rhs == Tensor("f", ["i", "a"])
    decoded_expression = decoder.decode_expression(encoded)
    assert decoded_expression.lhs == decoded.lhs
    assert decoded_expression.rhs == decoded.rhs


def test_l_tokens_can_decode_as_eom_left_amplitudes() -> None:
    decoder = PdaggerqDecoder(l_is_lambda=False)

    decoded = decoder.decode_node('[["l1(i)", "l1(a,i)", "l2(i,j,a,b)"]]')

    assert decoded == Mul(
        [
            Tensor("l1ip", ["i"]),
            Tensor("l1ee", ["a", "i"]),
            Tensor("l2ee", ["i", "j", "a", "b"]),
        ]
    )


def test_removes_disconnected_reference_energy_terms_from_eom_vectors() -> None:
    decoder = PdaggerqDecoder()
    encoded = (
        '[ ["+1.0", "f(i,i)", "r1(a)"], '
        '["-0.5", "<i,j||i,j>", "r1(a)"], '
        '["+1.0", "f(i,a)", "t1(a,i)", "r1(j)"], '
        '["+1.0", "<i,a||j,b>", "t2(a,b,i,j)", "r1(k)"], '
        '["+1.0", "<i,a||j,b>", "t1(a,i)", "t1(b,j)", "r1(k)"], '
        '["-1.0", "f(i,j)", "r1(j)"] ]'
    )

    decoded = decoder.decode_node(encoded)

    assert decoded == Mul([Scalar(-1.0), Tensor("f", ["i", "j"]), Tensor("r1ip", ["j"])])


def test_reference_energy_filter_treats_left_vectors_as_eom_only_when_requested() -> None:
    lambda_term = '[ ["+1.0", "f(j,j)", "l1(a,i)"] ]'

    assert PdaggerqDecoder().decode_node(lambda_term) == Mul([Tensor("f", ["j", "j"]), Tensor("l1", ["i", "a"])])
    assert PdaggerqDecoder(l_is_lambda=False).decode_node(lambda_term) == Scalar(0.0)


@pytest.mark.parametrize(
    ("encoded", "message"),
    [
        ("not serialized", "invalid pdaggerq serialization"),
        ("[1, 2]", "pdaggerq terms must be a list of token lists"),
        ('[["t4(a,b,c,d,i,j,k,l)"]]', "unsupported pdaggerq tensor"),
        ('[["<i,j|k>"]]', "expected 2 indices"),
        ('[["NaN"]]', "finite"),
    ],
)
def test_rejects_invalid_pdaggerq_data(encoded: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Node.decode(encoded, PdaggerqDecoder)


def test_expression_decoder_requires_lhs_and_terms() -> None:
    with pytest.raises(ValueError, match="lhs.*terms"):
        PdaggerqDecoder().decode_expression('[["+1.0", "f(i,j)"]]')
