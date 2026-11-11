import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from albert import (
    Add,
    CostModel,
    Expression,
    Metadata,
    Mul,
    PathRequest,
    PathRequests,
    Permutation,
    Scalar,
    ScalingLimits,
    ScalingMetric,
    Tensor,
    canonical_key,
    canonicalize,
    collect,
    common_subexpression_elimination,
    estimate_cost,
    evaluate_expression,
    expand,
    scaling_exponent,
    simplify,
    substitute_expressions,
)

DATA = Path(__file__).parent / "data" / "cse_regressions"
COST_MODEL = CostModel(multiply=1, add=1, read=0, write=0)


def decode(node: dict[str, Any]) -> Add | Mul | Scalar | Tensor:
    if node["kind"] == "scalar":
        return Scalar(node["value"])
    if node["kind"] == "tensor":
        return Tensor(node["symbol"], node["indices"])
    children = [decode(child) for child in node["children"]]
    return Add(children) if node["kind"] == "add" else Mul(children)


def load_cases(filename: str) -> dict[str, Any]:
    return json.loads((DATA / filename).read_text())


def metadata(case: dict[str, Any]) -> Metadata:
    return Metadata(
        indices=case["spaces"],
        sizes=case["sizes"],
        symmetries={
            symbol: [Permutation(item["permutation"], item["phase"]) for item in permutations]
            for symbol, permutations in case["symmetries"].items()
        },
    )


def total_cost(expressions: list[Expression], meta: Metadata) -> float:
    return sum(estimate_cost(expression.rhs, meta, COST_MODEL).total for expression in expressions)


def assert_cc_scaling(expressions: list[Expression], meta: Metadata) -> None:
    for spaces, expected in [(("o", "v"), 6), ("o", 4), ("v", 4)]:
        assert (
            max(scaling_exponent(expression, spaces, ScalingMetric.TIME, meta) for expression in expressions)
            == expected
        )


GREEDY_PATH_REQUESTS = PathRequests(bounds=(), requests=(PathRequest(greedy=1),))


def evaluation_arrays(expressions: list[Expression], meta: Metadata, seed: int) -> dict:
    index_spaces = {index: space for space, indices in meta.indices.items() for index in indices}
    space_sizes = {space: 2 for space in meta.indices}
    offsets = {}
    offset = 0
    for space, size in space_sizes.items():
        offsets[space] = slice(offset, offset + size)
        offset += size
    rng = np.random.default_rng(seed)
    full_arrays = {}
    arrays = {}
    for expression in expressions:
        for tensor in expression.rhs.search(lambda node: isinstance(node, Tensor)):
            spaces = tuple(index_spaces[index] for index in tensor.indices)
            rank_key = (tensor.symbol, len(spaces))
            if rank_key not in full_arrays:
                full = rng.standard_normal((offset,) * len(spaces))
                symmetries = meta.symmetries.get(tensor.symbol, [])
                if symmetries:
                    full = sum(
                        int(symmetry.phase) * np.transpose(full, symmetry.permutation) for symmetry in symmetries
                    ) / len(symmetries)
                full_arrays[rank_key] = full
            key = (tensor.symbol, spaces)
            if key not in arrays:
                arrays[key] = full_arrays[rank_key][tuple(offsets[space] for space in spaces)]
    return arrays


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


def test_rhf_ccsd_cse_uses_canonical_external_labels_for_shared_contractions() -> None:
    case = load_cases("rhf_cc.json")["RCCSD"]
    meta = metadata(case)
    expressions = [Expression(decode(item["lhs"]), decode(item["rhs"])) for item in case["expressions"]]

    result = common_subexpression_elimination(expressions, meta, cost_model=COST_MODEL)

    assert total_cost(result, meta) < 90_000_000_000


def test_ccd_cse_keeps_the_long_contraction_cost_below_regression_threshold() -> None:
    case = load_cases("rhf_cc.json")["CCD"]
    meta = metadata(case)
    expressions = [Expression(decode(item["lhs"]), decode(item["rhs"])) for item in case["expressions"]]

    result = common_subexpression_elimination(expressions, meta, cost_model=COST_MODEL)

    assert total_cost(result, meta) < 48_000_000_000


def test_ccd_terms_round_trip_through_cse_and_substitution() -> None:
    case = load_cases("rhf_cc.json")["CCD"]
    meta = metadata(case)
    expression = Expression(decode(case["expressions"][0]["lhs"]), decode(case["expressions"][0]["rhs"]))
    input_expression = Expression(expression.lhs, canonicalize(expand(expression.rhs), meta))

    result = common_subexpression_elimination([input_expression], meta, cost_model=COST_MODEL)
    assert len(result) > 1

    restored = substitute_expressions(result, meta)

    assert len(restored) == 1
    restored_rhs = canonicalize(expand(restored[0].rhs), meta)
    assert canonical_key(canonicalize(collect(restored_rhs), meta)) == canonical_key(
        canonicalize(collect(input_expression.rhs), meta)
    )


@pytest.mark.parametrize(
    ("case_name", "seed"),
    [("CCD", 17), ("RCCSD", 29)],
    ids=["ccd", "ccsd"],
)
def test_ccd_and_ccsd_evaluate_equally_after_canonicalization_simplification_and_cse(case_name: str, seed: int) -> None:
    case = load_cases("rhf_cc.json")[case_name]
    meta = metadata(case)
    small_meta = Metadata(
        indices=meta.indices,
        sizes={space: 2 for space in meta.indices},
        symmetries=meta.symmetries,
    )
    source = case["expressions"][0]
    expression = Expression(decode(source["lhs"]), decode(source["rhs"]))
    canonical = Expression(expression.lhs, canonicalize(expand(expression.rhs), small_meta))
    simplified = Expression(canonical.lhs, simplify(canonical.rhs))
    arrays = evaluation_arrays([expression, canonical, simplified], small_meta, seed)
    factored = common_subexpression_elimination([simplified], meta, cost_model=COST_MODEL)

    expected = evaluate_expression([expression], arrays, small_meta)
    for stage in (canonical, simplified):
        np.testing.assert_allclose(evaluate_expression([stage], arrays, small_meta), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(evaluate_expression(factored, arrays, small_meta), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("case_name", ["CCD", "RCCSD"], ids=["ccd", "ccsd"])
def test_ccd_and_ccsd_cse_output_has_expected_scaling(case_name: str) -> None:
    case = load_cases("rhf_cc.json")[case_name]
    meta = metadata(case)
    expressions = [Expression(decode(item["lhs"]), decode(item["rhs"])) for item in case["expressions"]]

    factored = common_subexpression_elimination(
        expressions,
        meta,
        cost_model=COST_MODEL,
        path_requests=GREEDY_PATH_REQUESTS,
    )

    assert_cc_scaling(factored, meta)


def test_df_tt_cse_respects_scaling_limits_without_losing_flop_performance() -> None:
    case = load_cases("df_tt.json")
    meta = metadata(case)
    limits = ScalingLimits(
        time={tuple(item["spaces"]): item["limit"] for item in case["time_limits"]},
        memory={tuple(item["spaces"]): item["limit"] for item in case["memory_limits"]},
    )
    expression = Expression(decode(case["lhs"]), decode(case["rhs"]))

    result = common_subexpression_elimination([expression], meta, limits, COST_MODEL)

    assert total_cost(result, meta) < 219_000_000_000
    for optimized in result:
        for spaces, maximum in limits.time.items():
            assert scaling_exponent(optimized, spaces, ScalingMetric.TIME, meta) <= maximum
        for spaces, maximum in limits.memory.items():
            assert scaling_exponent(optimized, spaces, ScalingMetric.MEMORY, meta) <= maximum
