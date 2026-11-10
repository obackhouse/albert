"""Long contraction regressions with checked-in, standalone input expressions."""

import json
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
        spaces=case["spaces"],
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
        assert max(scaling_exponent(expression, spaces, "time", meta) for expression in expressions) == expected


GREEDY_PATH_REQUESTS = PathRequests(bounds=(), requests=(PathRequest(greedy=1),))


def evaluation_arrays(expressions: list[Expression], meta: Metadata, seed: int) -> dict:
    index_spaces = {index: space for space, indices in meta.spaces.items() for index in indices}
    space_sizes = {space: 2 for space in meta.spaces}
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
        spaces=meta.spaces,
        sizes={space: 2 for space in meta.spaces},
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
            assert scaling_exponent(optimized, spaces, "time", meta) <= maximum
        for spaces, maximum in limits.memory.items():
            assert scaling_exponent(optimized, spaces, "memory", meta) <= maximum
