from itertools import product
from pathlib import Path

import numpy as np

from albert import (
    CostModel,
    Expression,
    ScalingMetric,
    common_subexpression_elimination,
    evaluate_expression,
    scaling_exponent,
    simplify,
)
from albert.chemistry import PdaggerqDecoder
from albert.chemistry.metadata import Metadata
from albert.chemistry.options import ReferenceType
from albert.chemistry.spin import spin_integrate

DATA = Path(__file__).parent / "data"
PYSCF_LIH_STO3G_MP2_CORRELATION_ENERGY = -0.012904096452760878


def _mp2_arrays() -> tuple[dict[tuple[str, tuple[str, ...]], np.ndarray], Metadata]:
    with np.load(DATA / "lih_sto3g.npz") as data:
        orbital_energies = data["orbital_energies"]
        chemist_eris = data["mo_eri_chemist"]
        n_occupied_spatial = int(data["nocc"])

    # Convert chemist order to physicist order, v[p,q,r,s] = (p r | q s).
    physicist_eris = chemist_eris.transpose(0, 2, 1, 3)
    n_orbitals = len(orbital_energies)
    occupied = slice(0, n_occupied_spatial)
    virtual = slice(n_occupied_spatial, n_orbitals)
    occupied_energies = orbital_energies[occupied]
    virtual_energies = orbital_energies[virtual]
    denominators = (
        occupied_energies[:, None, None, None]
        + occupied_energies[None, :, None, None]
        - virtual_energies[None, None, :, None]
        - virtual_energies[None, None, None, :]
    )
    spatial_v_oovv = physicist_eris[occupied, occupied, virtual, virtual]
    t2 = spatial_v_oovv / denominators
    arrays = {("t2", ("o", "o", "v", "v")): t2}
    slices = {"o": occupied, "v": virtual}
    for spaces in product(("o", "v"), repeat=4):
        arrays[("v", spaces)] = physicist_eris[tuple(slices[space] for space in spaces)]
    metadata = Metadata(
        indices={"o": ["i", "j"], "v": ["a", "b"]},
        sizes={"o": n_occupied_spatial, "v": n_orbitals - n_occupied_spatial},
    )
    return arrays, metadata


def test_pdaggerq_mp2_energy_pipeline_matches_pyscf_lih_sto3g() -> None:
    decoder = PdaggerqDecoder()
    expression = decoder.decode_expression((DATA / "mp2_energy.json").read_text())
    arrays, metadata = _mp2_arrays()
    assert {key for key in arrays if key[0] == "t2"} == {("t2", ("o", "o", "v", "v"))}

    rhf_expressions = spin_integrate(expression, ReferenceType.GHF, ReferenceType.RHF, metadata)
    simplified = [Expression(item.lhs, simplify(item.rhs)) for item in rhf_expressions]
    factored = common_subexpression_elimination(
        simplified,
        metadata,
        cost_model=CostModel(multiply=1, add=1, read=0, write=0),
    )
    evaluated = evaluate_expression(factored, arrays, metadata)

    np.testing.assert_allclose(
        evaluated,
        np.asarray(PYSCF_LIH_STO3G_MP2_CORRELATION_ENERGY),
        rtol=1e-11,
        atol=1e-12,
    )

    assert max(scaling_exponent(item, ("o", "v"), ScalingMetric.TIME, metadata) for item in factored) == 4
    assert max(scaling_exponent(item, "o", ScalingMetric.TIME, metadata) for item in factored) == 2
    assert max(scaling_exponent(item, "v", ScalingMetric.TIME, metadata) for item in factored) == 2
