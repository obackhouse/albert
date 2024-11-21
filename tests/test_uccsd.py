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
from albert.qc.spin import ghf_to_uhf
from albert.tensor import Tensor


def _kwargs(strategy, transposes, greedy_cutoff, drop_cutoff):
    return {
        "strategy": strategy,
        "transposes": transposes,
        "greedy_cutoff": greedy_cutoff,
        "drop_cutoff": drop_cutoff,
    }


@pytest.mark.parametrize(
    "optimise, canonicalise, kwargs",
    [
        (False, False, _kwargs(None, None, None, None)),
        (True, False, _kwargs("greedy", "ignore", -1, 2)),
        (True, True, _kwargs("greedy", "ignore", 2, 2)),
    ],
)
def test_uccsd_einsum(helper, optimise, canonicalise, kwargs):
    with open(f"{os.path.dirname(__file__)}/_test_uccsd.py", "w") as file:
        try:
            _test_uccsd_einsum(helper, file, optimise, canonicalise, kwargs)
        except Exception as e:
            raise e
        finally:
            os.remove(f"{os.path.dirname(__file__)}/_test_uccsd.py")


def _test_uccsd_einsum(helper, file, optimise, canonicalise, kwargs):
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
    energy = ghf_to_uhf(energy)
    if canonicalise:
        energy = tuple(e.canonicalise(indices=True).collect() for e in energy)
    output = tuple(Tensor(name="e_cc") for _ in energy)

    if optimise:
        output_expr = _optimise(output, energy, **kwargs)
    else:
        output_expr = list(zip(output, energy))

    codegen(
        "energy",
        output,
        output_expr,
    )

    pq.clear()
    pq.set_left_operators([["e1(i,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    t1 = pq.fully_contracted_strings()
    t1 = import_from_pdaggerq(t1)
    t1 = ghf_to_uhf(t1)
    if canonicalise:
        t1 = tuple(t.canonicalise(indices=True).collect() for t in t1)
    output_t1 = tuple(
        Tensor(*sorted(t.external_indices, key=lambda i: "ijab".index(i.name)), name=f"t1new")
        for i, t in enumerate(t1)
    )

    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    t2 = pq.fully_contracted_strings()
    t2_expr = tuple()
    for spins in ("aaaa", "abab", "baba", "bbbb"):
        index_spins = dict(zip("ijab", spins))
        t2_expr += ghf_to_uhf(import_from_pdaggerq(t2, index_spins=index_spins))
    t2 = t2_expr
    if canonicalise:
        t2 = tuple(t.canonicalise(indices=True).collect() for t in t2)
    output_t2 = tuple(
        Tensor(*sorted(t.external_indices, key=lambda i: "ijab".index(i.name)), name=f"t2new")
        for i, t in enumerate(t2)
    )

    outputs = output_t1 + output_t2
    if optimise:
        output_expr = _optimise(outputs, t1 + t2, **kwargs)
    else:
        output_expr = list(zip(outputs, t1 + t2))

    codegen(
        "update_amplitudes",
        outputs,
        output_expr,
        as_dict=True,
    )

    module = importlib.import_module(f"_test_uccsd")
    energy = module.energy
    update_amplitudes = module.update_amplitudes

    mol = gto.M(atom="Be 0 0 0; H 0 0 1.5", basis="cc-pvdz", verbose=0, spin=1)
    mf = scf.UHF(mol).run()
    ccsd = cc.CCSD(mf)
    ccsd.max_cycle = 3
    ccsd.diis = False
    ccsd.kernel()

    eo = (mf.mo_energy[0][mf.mo_occ[0] > 0], mf.mo_energy[1][mf.mo_occ[1] > 0])
    ev = (mf.mo_energy[0][mf.mo_occ[0] == 0], mf.mo_energy[1][mf.mo_occ[1] == 0])
    f = SimpleNamespace()
    f.aa = SimpleNamespace()
    f.bb = SimpleNamespace()
    f.aa.oo = np.diag(eo[0])
    f.aa.vv = np.diag(ev[0])
    f.bb.oo = np.diag(eo[1])
    f.bb.vv = np.diag(ev[1])
    f.aa.ov = np.zeros((eo[0].size, ev[0].size))
    f.aa.vo = np.zeros((ev[0].size, eo[0].size))
    f.bb.ov = np.zeros((eo[1].size, ev[1].size))
    f.bb.vo = np.zeros((ev[1].size, eo[1].size))

    co = (mf.mo_coeff[0][:, mf.mo_occ[0] > 0], mf.mo_coeff[1][:, mf.mo_occ[1] > 0])
    cv = (mf.mo_coeff[0][:, mf.mo_occ[0] == 0], mf.mo_coeff[1][:, mf.mo_occ[1] == 0])
    v = SimpleNamespace()
    for spin1 in "ab":
        for spin2 in "ab":
            setattr(v, spin1 * 2 + spin2 * 2, SimpleNamespace())
            for key in itertools.product("ov", repeat=4):
                coeffs = tuple(
                    co["ab".index(s)] if k == "o" else cv["ab".index(s)]
                    for k, s in zip(key, spin1 * 2 + spin2 * 2)
                )
                shape = tuple(c.shape[-1] for c in coeffs)
                v_key = ao2mo.kernel(mol, coeffs, compact=False).reshape(shape)
                setattr(getattr(v, spin1 * 2 + spin2 * 2), "".join(key), v_key)

    t1 = SimpleNamespace(aa=SimpleNamespace(ov=ccsd.t1[0]), bb=SimpleNamespace(ov=ccsd.t1[1]))
    t2 = SimpleNamespace(
        aaaa=SimpleNamespace(oovv=ccsd.t2[0]),
        abab=SimpleNamespace(oovv=ccsd.t2[1]),
        bbbb=SimpleNamespace(oovv=ccsd.t2[2]),
    )

    e1 = np.ravel(energy(f=f, v=v, t1=t1, t2=t2)).item()
    e2 = ccsd.energy(
        t1=(t1.aa.ov, t1.bb.ov),
        # Note different representation of t2 compared to pyscf
        t2=(
            t2.aaaa.oovv - t2.aaaa.oovv.swapaxes(2, 3),
            t2.abab.oovv,
            t2.bbbb.oovv - t2.bbbb.oovv.swapaxes(2, 3),
        ),
    )
    assert np.allclose(e1, e2)

    d = (eo[0][:, None] - ev[0][None, :], eo[1][:, None] - ev[1][None, :])
    amps = update_amplitudes(f=f, v=v, t1=t1, t2=t2)
    amps["t1new"].aa.ov = amps["t1new"].aa.ov / d[0] + t1.aa.ov
    amps["t1new"].bb.ov = amps["t1new"].bb.ov / d[1] + t1.bb.ov
    amps["t2new"].aaaa.oovv = (
        amps["t2new"].aaaa.oovv / (d[0][:, None, :, None] + d[0][None, :, None, :]) + t2.aaaa.oovv
    )
    amps["t2new"].abab.oovv = (
        amps["t2new"].abab.oovv / (d[0][:, None, :, None] + d[1][None, :, None, :]) + t2.abab.oovv
    )
    amps["t2new"].bbbb.oovv = (
        amps["t2new"].bbbb.oovv / (d[1][:, None, :, None] + d[1][None, :, None, :]) + t2.bbbb.oovv
    )
    e1 = np.ravel(energy(f=f, v=v, t1=amps["t1new"], t2=amps["t2new"])).item()
    e2 = ccsd.energy(
        t1=(amps["t1new"].aa.ov, amps["t1new"].bb.ov),
        # Note different representation of t2 compared to pyscf
        t2=(
            amps["t2new"].aaaa.oovv - amps["t2new"].aaaa.oovv.swapaxes(2, 3),
            amps["t2new"].abab.oovv,
            amps["t2new"].bbbb.oovv - amps["t2new"].bbbb.oovv.swapaxes(2, 3),
        ),
    )
    assert np.allclose(e1, e2)
