import importlib
import itertools
import os

import numpy as np
import pdaggerq
import pytest

from albert.code.einsum import EinsumCodeGenerator
from albert.misc import ExclusionSet
from albert.opt._gristmill import optimise_gristmill
from albert.qc._pdaggerq import import_from_pdaggerq, remove_reference_energy
from albert.qc.spin import ghf_to_rhf
from albert.tensor import Tensor


@pytest.mark.parametrize(
    "optimise, strategy, transposes, greedy_cutoff, drop_cutoff",
    [
        (False, None, None, None, None),
        (True, "trav", "natural", -1, -1),
        (True, "opt", "natural", -1, -1),
        (True, "greedy", "ignore", -1, 2),
        (True, "greedy", "ignore", 2, 2),
    ],
)
def test_rccsd_einsum(helper, optimise, strategy, transposes, greedy_cutoff, drop_cutoff):
    with open(f"{os.path.dirname(__file__)}/_test_rccsd.py", "w") as file:
        try:
            _test_rccsd_einsum(
                helper, file, optimise, strategy, transposes, greedy_cutoff, drop_cutoff
            )
        except Exception as e:
            raise e
        finally:
            os.remove(f"{os.path.dirname(__file__)}/_test_rccsd.py")


def _test_rccsd_einsum(helper, file, optimise, strategy, transposes, greedy_cutoff, drop_cutoff):
    class _EinsumCodeGenerator(EinsumCodeGenerator):
        _add_spaces = ExclusionSet(("t1", "t2"))

    codegen = _EinsumCodeGenerator(stdout=file)
    codegen.preamble()

    pq = pdaggerq.pq_helper("fermi")

    pq.clear()
    pq.set_left_operators([["1"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    energy = pq.fully_contracted_strings()
    energy = remove_reference_energy(energy)
    energy = import_from_pdaggerq(energy)
    energy = ghf_to_rhf(energy).collect()
    output = Tensor(name="e_cc")
    output_expr = [(output, energy)]

    codegen(
        "energy",
        [output],
        output_expr,
    )

    pq.clear()
    pq.set_left_operators([["e1(i,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    t1 = pq.fully_contracted_strings()
    t1 = import_from_pdaggerq(t1)
    t1 = ghf_to_rhf(t1).collect()
    output_t1 = Tensor(*t1.external_indices, name="t1new")

    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    t2 = pq.fully_contracted_strings()
    t2 = import_from_pdaggerq(t2)
    t2 = ghf_to_rhf(t2).collect()
    output_t2 = Tensor(*t2.external_indices, name="t2new")

    outputs = [output_t1, output_t2]
    if optimise:
        output_expr = optimise_gristmill(
            outputs,
            [t1, t2],
            strategy=strategy,
            transposes=transposes,
            greedy_cutoff=greedy_cutoff,
            drop_cutoff=drop_cutoff,
        )
    else:
        output_expr = [(output_t1, t1), (output_t2, t2)]

    codegen(
        "update_amplitudes",
        outputs,
        output_expr,
        as_dict=True,
    )

    module = importlib.import_module(f"_test_rccsd")
    energy = module.energy
    update_amplitudes = module.update_amplitudes

    nocc = 4
    nvir = 16

    f = lambda: None
    f.oo = np.diag(helper.random((nocc,), seed=123))
    f.ov = np.zeros((nocc, nvir))
    f.vo = np.zeros((nvir, nocc))
    f.vv = np.diag(helper.random((nvir,), seed=234))

    v = lambda: None
    v_full = helper.random((nocc + nvir,) * 4, seed=345)
    v_full = 0.5 * (v_full + v_full.transpose(1, 0, 2, 3))
    v_full = 0.5 * (v_full + v_full.transpose(0, 1, 3, 2))
    v_full = 0.5 * (v_full + v_full.transpose(2, 3, 0, 1))
    for key in itertools.product("ov", repeat=4):
        slices = tuple(slice(None, nocc) if k == "o" else slice(nocc, None) for k in key)
        setattr(v, "".join(key), v_full[slices])

    t1 = helper.random((nocc, nvir), seed=456)
    t2 = v.oovv.copy()

    amps = update_amplitudes(f=f, v=v, t1=t1, t2=t2)
    e_cc = energy(f=f, v=v, t1=amps["t1new"], t2=amps["t2new"])

    assert np.allclose(e_cc, 32450.175915553173)
