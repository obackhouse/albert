import numpy as np
import pytest

from albert import (
    Add,
    CostModel,
    Expression,
    Metadata,
    Mul,
    Scalar,
    Tensor,
    canonicalize,
    common_subexpression_elimination,
    evaluate_expression,
    simplify,
)


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.complex64, np.complex128],
    ids=["float32", "float64", "complex64", "complex128"],
)
def test_evaluates_contraction_for_supported_dtypes(dtype) -> None:
    a = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    b = np.asarray([2, 3, 4], dtype=dtype)
    expression = Expression(
        Tensor("Y", ["i"]),
        Mul([Tensor("A", ["i", "k"]), Tensor("B", ["k"])]),
    )

    result = evaluate_expression(
        [expression],
        {("A", ("I", "K")): a, ("B", ("K",)): b},
        Metadata(spaces={"I": ["i"], "K": ["k"]}, sizes={"I": 2, "K": 3}),
    )

    np.testing.assert_array_equal(result, np.asarray([20, 47], dtype=dtype))
    assert result.dtype == np.dtype(dtype)


def test_evaluates_expression_dependencies_in_any_input_order() -> None:
    contraction = Expression(
        Tensor("U", []),
        Mul([Tensor("A", ["i", "k"]), Tensor("B", ["i", "k"])]),
    )
    output = Expression(
        Tensor("Y", ["j"]),
        Mul([Tensor("U", []), Tensor("C", ["j"])]),
    )
    arrays = {
        ("A", ("I", "K")): np.asarray([[1, 2], [3, 4]], dtype=np.float64),
        ("B", ("I", "K")): np.asarray([[5, 6], [7, 8]], dtype=np.float64),
        ("C", ("J",)): np.asarray([2, 3, 4], dtype=np.float64),
    }

    result = evaluate_expression(
        [output, contraction],
        arrays,
        Metadata(
            spaces={"I": ["i"], "J": ["j"], "K": ["k"]},
            sizes={"I": 2, "J": 3, "K": 2},
        ),
    )

    np.testing.assert_array_equal(result, np.asarray([140, 210, 280]))


def test_same_symbol_can_use_distinct_arrays_for_distinct_spaces() -> None:
    expression = Expression(
        Tensor("Y", ["i", "a"]),
        Mul([Tensor("T", ["i"]), Tensor("T", ["a"])]),
    )
    occupied = np.asarray([2, 3], dtype=np.float64)
    virtual = np.asarray([5, 7, 11], dtype=np.float64)

    result = evaluate_expression(
        [expression],
        {("T", ("O",)): occupied, ("T", ("V",)): virtual},
        Metadata(spaces={"O": ["i"], "V": ["a"]}, sizes={"O": 2, "V": 3}),
    )

    np.testing.assert_array_equal(result, np.outer(occupied, virtual))


def test_evaluates_additions_and_scalar_coefficients() -> None:
    expression = Expression(
        Tensor("Y", ["i"]),
        Add(
            [
                Tensor("A", ["i"]),
                Mul([Scalar(2), Tensor("B", ["i"])]),
            ]
        ),
    )
    arrays = {
        ("A", ("I",)): np.asarray([1, 2], dtype=np.float32),
        ("B", ("I",)): np.asarray([3, 4], dtype=np.float32),
    }

    result = evaluate_expression([expression], arrays, Metadata(spaces={"I": ["i"]}, sizes={"I": 2}))

    np.testing.assert_array_equal(result, np.asarray([7, 10], dtype=np.float32))


def test_rejects_missing_array_for_tensor_and_wrong_space_shape() -> None:
    expression = Expression(Tensor("Y", ["i"]), Tensor("A", ["i"]))
    metadata = Metadata(spaces={"I": ["i"]}, sizes={"I": 2})

    with pytest.raises(ValueError, match="no array supplied"):
        evaluate_expression([expression], {}, metadata)
    with pytest.raises(ValueError, match="declared size"):
        evaluate_expression([expression], {("A", ("I",)): np.ones(3)}, metadata)


def test_rejects_multiple_terminal_outputs_and_mixed_dtypes() -> None:
    first = Expression(Tensor("Y", ["i"]), Tensor("A", ["i"]))
    second = Expression(Tensor("Z", ["i"]), Tensor("B", ["i"]))
    metadata = Metadata(spaces={"I": ["i"]}, sizes={"I": 2})

    with pytest.raises(ValueError, match="exactly one terminal output"):
        evaluate_expression(
            [first, second],
            {
                ("A", ("I",)): np.ones(2),
                ("B", ("I",)): np.ones(2),
            },
            metadata,
        )
    with pytest.raises(TypeError, match="same dtype"):
        evaluate_expression(
            [first],
            {
                ("A", ("I",)): np.ones(2, dtype=np.float32),
                ("unused", ()): np.ones(2, dtype=np.float64),
            },
            metadata,
        )


def test_requires_metadata() -> None:
    expression = Expression(Tensor("Y", ["i"]), Tensor("A", ["i"]))
    with pytest.raises(TypeError, match="metadata is required"):
        evaluate_expression([expression], {("A", ("I",)): np.ones(2)}, None)


def test_canonicalize_simplify_and_cse_preserve_evaluation() -> None:
    metadata = Metadata(
        spaces={"I": ["i"], "J": ["j"], "K": ["k"], "L": ["l"]},
        sizes={"I": 2, "J": 2, "K": 2, "L": 2},
    )
    a = Tensor("A", ["i", "k"])
    b = Tensor("B", ["k", "j"])
    c = Tensor("C", ["j", "l"])
    d = Tensor("D", ["j", "l"])
    original = Expression(Tensor("Y", ["i", "l"]), Add([Mul([a, b, c]), Mul([a, b, d])]))
    arrays = {
        ("A", ("I", "K")): np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        ("B", ("K", "J")): np.asarray([[2.0, 1.0], [0.0, 3.0]]),
        ("C", ("J", "L")): np.asarray([[1.0, 2.0], [3.0, 0.0]]),
        ("D", ("J", "L")): np.asarray([[0.0, 1.0], [2.0, 4.0]]),
    }
    canonical = Expression(original.lhs, canonicalize(original.rhs, metadata))
    simplified = Expression(canonical.lhs, simplify(canonical.rhs))
    factored = common_subexpression_elimination(
        [simplified], metadata, cost_model=CostModel(multiply=1, add=0, read=0, write=0)
    )

    expected = evaluate_expression([original], arrays, metadata)
    for stage in (canonical, simplified):
        np.testing.assert_allclose(evaluate_expression([stage], arrays, metadata), expected)
    assert len(factored) > 1
    np.testing.assert_allclose(evaluate_expression(factored, arrays, metadata), expected)
