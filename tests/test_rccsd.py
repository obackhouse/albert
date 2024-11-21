import importlib
import itertools
import os
from types import SimpleNamespace

import numpy as np
import pdaggerq
import pytest
from pyscf import ao2mo, cc, gto, scf

from albert.code.einsum import EinsumCodeGenerator
from albert.opt import optimise as _optimise
from albert.qc._pdaggerq import import_from_pdaggerq, remove_reference_energy
from albert.qc.spin import ghf_to_rhf
from albert.tensor import Tensor


def _kwargs(strategy, transposes, greedy_cutoff, drop_cutoff):
    return {
        "strategy": strategy,
        "transposes": transposes,
        "greedy_cutoff": greedy_cutoff,
        "drop_cutoff": drop_cutoff,
    }


@pytest.mark.parametrize(
    "optimise, method, canonicalise, kwargs",
    [
        (False, None, False, _kwargs(None, None, None, None)),
        (True, "gristmill", True, _kwargs("trav", "natural", -1, -1)),
        (True, "gristmill", True, _kwargs("opt", "natural", -1, -1)),
        (True, "gristmill", False, _kwargs("greedy", "ignore", -1, 2)),
        (True, "gristmill", True, _kwargs("greedy", "ignore", 2, 2)),
    ],
)
def test_rccsd_einsum(helper, optimise, method, canonicalise, kwargs):
    with open(f"{os.path.dirname(__file__)}/_test_rccsd.py", "w") as file:
        try:
            _test_rccsd_einsum(helper, file, optimise, method, canonicalise, kwargs)
        except Exception as e:
            raise e
        finally:
            os.remove(f"{os.path.dirname(__file__)}/_test_rccsd.py")


def _test_rccsd_einsum(helper, file, optimise, method, canonicalise, kwargs):
    codegen = EinsumCodeGenerator(stdout=file)
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
    energy = ghf_to_rhf(energy)
    if canonicalise:
        energy = energy.canonicalise(indices=True).collect()
    output = Tensor(name="e_cc")

    if optimise:
        output_expr = _optimise([output], [energy], method=method, **kwargs)
    else:
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
    t1 = import_from_pdaggerq(t1, index_spins=dict(i="a", a="a"))
    t1 = ghf_to_rhf(t1)
    if canonicalise:
        t1 = t1.canonicalise(indices=True).collect()
    output_t1 = Tensor(
        *sorted(t1.external_indices, key=lambda i: "ijab".index(i.name)), name="t1new"
    )

    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    t2 = pq.fully_contracted_strings()
    t2 = import_from_pdaggerq(t2, index_spins=dict(i="a", j="b", a="a", b="b"))
    t2 = ghf_to_rhf(t2)
    if canonicalise:
        t2 = t2.canonicalise(indices=True).collect()
    output_t2 = Tensor(
        *sorted(t2.external_indices, key=lambda i: "ijab".index(i.name)), name="t2new"
    )

    outputs = [output_t1, output_t2]
    if optimise:
        output_expr = _optimise(
            outputs,
            [t1, t2],
            method=method,
            **kwargs,
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

    mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).run()
    ccsd = cc.CCSD(mf)
    ccsd.max_cycle = 3
    ccsd.diis = False
    ccsd.kernel()

    eo = mf.mo_energy[mf.mo_occ > 0]
    ev = mf.mo_energy[mf.mo_occ == 0]
    f = SimpleNamespace()
    f.oo = np.diag(eo)
    f.vv = np.diag(ev)
    f.ov = np.zeros((eo.size, ev.size))
    f.vo = np.zeros((ev.size, eo.size))

    co = mf.mo_coeff[:, mf.mo_occ > 0]
    cv = mf.mo_coeff[:, mf.mo_occ == 0]
    v = SimpleNamespace()
    for key in itertools.product("ov", repeat=4):
        coeffs = tuple(co if k == "o" else cv for k in key)
        shape = tuple(c.shape[-1] for c in coeffs)
        v_key = ao2mo.kernel(mol, coeffs, compact=False).reshape(shape)
        setattr(v, "".join(key), v_key)

    t1 = ccsd.t1
    t2 = ccsd.t2

    e1 = np.ravel(energy(f=f, v=v, t1=SimpleNamespace(ov=t1), t2=SimpleNamespace(oovv=t2))).item()
    e2 = ccsd.energy(t1=t1, t2=t2)
    assert np.allclose(e1, e2)

    d = eo[:, None] - ev[None, :]
    amps = update_amplitudes(f=f, v=v, t1=SimpleNamespace(ov=t1), t2=SimpleNamespace(oovv=t2))
    amps["t1new"].ov = amps["t1new"].ov / d + t1
    amps["t2new"].oovv = amps["t2new"].oovv / (d[:, None, :, None] + d[None, :, None, :]) + t2
    e1 = ccsd.energy(*ccsd.update_amps(t1, t2, ccsd.ao2mo()))
    e2 = ccsd.energy(t1=amps["t1new"].ov, t2=amps["t2new"].oovv)
    assert np.allclose(e1, e2)
