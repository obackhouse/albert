"""Long contraction regressions with checked-in, standalone input expressions."""

import json
from pathlib import Path
from typing import Any

from albert import (
    Add,
    CostModel,
    Expression,
    Metadata,
    Mul,
    Permutation,
    Scalar,
    ScalingLimits,
    Tensor,
    common_subexpression_elimination,
    estimate_cost,
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
            assert optimized.rhs.scaling_exponent(spaces, "time", meta) <= maximum
        for spaces, maximum in limits.memory.items():
            assert optimized.rhs.scaling_exponent(spaces, "memory", meta) <= maximum
