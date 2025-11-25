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
from albert.qc._pdaggerq import remove_reference_energy
from albert.qc import import_expression, adapt_spin
from albert.tensor import Tensor
from albert.expression import Expression


def _kwargs(strategy, transposes, greedy_cutoff, drop_cutoff):
    return {
        "strategy": strategy,
        "transposes": transposes,
        "greedy_cutoff": greedy_cutoff,
        "drop_cutoff": drop_cutoff,
    }


@pytest.mark.parametrize(
    "optimise, method, kwargs",
    [
        (False, None, _kwargs(None, None, None, None)),
        (True, "gristmill", _kwargs("trav", "natural", -1, -1)),
        (True, "gristmill", _kwargs("greedy", "ignore", -1, 2)),
        (True, "gristmill", _kwargs("greedy", "ignore", 2, 2)),
        (True, "albert", {}),
    ],
)
def test_rccsd_einsum(helper, optimise, method, kwargs):
    with open(f"{os.path.dirname(__file__)}/_test_rccsd.py", "w") as file:
        try:
            _test_rccsd_einsum(helper, file, optimise, method, kwargs)
        except Exception as e:
            raise e
        finally:
            os.remove(f"{os.path.dirname(__file__)}/_test_rccsd.py")


def _test_rccsd_einsum(helper, file, optimise, method, kwargs):
    codegen = EinsumCodeGenerator(stdout=file)
    codegen.preamble()

    pq = pdaggerq.pq_helper("fermi")

    pq.clear()
    pq.set_left_operators([["1"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    energy = pq.strings()
    energy = remove_reference_energy(energy)
    energy = import_expression(energy, package="pdaggerq", name="e_cc")
    exprs = adapt_spin(energy, target_spin="rhf")
    if optimise:
        exprs = _optimise(exprs, method=method, **kwargs)

    codegen("energy", [expr.lhs for expr in exprs], exprs)

    pq.clear()
    pq.set_left_operators([["e1(i,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    t1 = pq.strings()
    t1 = import_expression(t1, package="pdaggerq", index_spins=dict(i="a", a="a"), name="t1new")
    t1 = adapt_spin(t1, target_spin="rhf")

    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    t2 = pq.strings()
    t2 = import_expression(t2, package="pdaggerq", index_spins=dict(i="a", j="b", a="a", b="b"), name="t2new")
    t2 = adapt_spin(t2, target_spin="rhf")

    exprs = t1 + t2
    if optimise:
        exprs = _optimise(exprs, method=method, **kwargs)

    codegen("update_amplitudes", [expr.lhs for expr in exprs], exprs, as_dict=True)

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
