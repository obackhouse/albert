"""Test the generation of RCCSD.
"""

import os
import unittest
import warnings
from io import StringIO

import pdaggerq
import numpy as np

from albert.codegen.einsum import EinsumCodeGen
from albert.qc._pdaggerq import import_from_pdaggerq
from albert.canon import canonicalise_indices
from albert.tensor import Tensor
from albert.qc.spin import generalised_to_restricted
from albert.optim._gristmill import optimise

np.random.seed(1)


def name_generator(tensor):
    if tensor.name in ("f", "v"):
        spaces = ["o" if i in "ijklmn" else "v" for i in tensor.indices]
        return f"{tensor.name}.{''.join(spaces)}"
    else:
        return tensor.name


class TestRCCSD(unittest.TestCase):
    def generate_code_pdaggerq(self):
        stdout = StringIO()
        codegen = EinsumCodeGen(
            stdout=stdout,
            name_generator=name_generator,
        )

        pq = pdaggerq.pq_helper("fermi")

        pq.clear()
        pq.set_left_operators([["1"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
        pq.simplify()
        energy = pq.fully_contracted_strings()
        energy = [e for e in energy if e[-1] not in ("f(i,i)", "<j,i||j,i>")]
        output_energy = Tensor(name="e_cc")

        codegen(
            "energy",
            [output_energy],
            [output_energy],
            [generalised_to_restricted(canonicalise_indices(import_from_pdaggerq(energy), "ijklmn", "abcdef"))],
        )

        pq.clear()
        pq.set_left_operators([["e1(i,a)"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
        pq.simplify()
        t1 = pq.fully_contracted_strings()

        pq.clear()
        pq.set_left_operators([["e2(i,j,b,a)"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
        pq.simplify()
        t2 = pq.fully_contracted_strings()

        expr_t1 = import_from_pdaggerq(t1, index_spins={"i": "α", "a": "α"})
        expr_t1 = canonicalise_indices(expr_t1, "ijklmn", "abcdef")
        expr_t1 = generalised_to_restricted(expr_t1)
        output_t1 = Tensor("i", "a", name="t1new")

        expr_t2 = import_from_pdaggerq(t2, index_spins={"i": "α", "j": "β", "a": "α", "b": "β"})
        expr_t2 = canonicalise_indices(expr_t2, "ijklmn", "abcdef")
        expr_t2 = generalised_to_restricted(expr_t2)
        output_t2 = Tensor("i", "j", "a", "b", name="t2new")

        returns = (
            Tensor("i", "a", name="t1new"),
            Tensor("i", "j", "a", "b", name="t2new"),
        )

        opt = optimise(
            (output_t1, expr_t1),
            (output_t2, expr_t2),
            index_groups=["ijklmn", "abcdef"],
            sizes={
                **{i: 4 for i in "ijklmn"},
                **{i: 20 for i in "abcdef"},
            },
            strategy="greedy",
        )
        outputs, exprs = zip(*opt)

        codegen(
            "update_amps",
            returns,
            outputs,
            exprs,
            as_dict=True,
        )

        return stdout

    def test_rccsd(self):
        stdout = self.generate_code_pdaggerq()
        exec(stdout.getvalue(), globals())
        stdout.close()

        nocc = 4
        nvir = 20

        f = lambda: None
        f.oo = np.diag(np.random.random(nocc))
        f.ov = np.zeros((nocc, nvir))
        f.vo = np.zeros((nvir, nocc))
        f.vv = np.diag(np.random.random(nvir))

        v = lambda: None
        v_full = np.random.random((nocc + nvir,) * 4)
        v_full = v_full + v_full.transpose(0, 1, 3, 2) + v_full.transpose(1, 0, 2, 3) + v_full.transpose(1, 0, 3, 2)
        v_full = v_full + v_full.transpose(2, 3, 0, 1)
        v.oooo = v_full[:nocc, :nocc, :nocc, :nocc]
        v.ovov = v_full[:nocc, nocc:, :nocc, nocc:]
        v.vvvv = v_full[nocc:, nocc:, nocc:, nocc:]
        v.oovv = v_full[:nocc, :nocc, nocc:, nocc:]
        v.ooov = v_full[:nocc, :nocc, :nocc, nocc:]
        v.ovvv = v_full[:nocc, nocc:, nocc:, nocc:]
        v.vovo = v_full[nocc:, :nocc, nocc:, :nocc]

        t1 = np.random.random((nocc, nvir))
        t2 = v.oovv

        amps = update_amps(f=f, v=v, t1=t1, t2=t2)
        e_cc = energy(f=f, v=v, t1=amps["t1new"], t2=amps["t2new"])

        self.assertAlmostEqual(e_cc, 23956467724253, places=-1)


if __name__ == "__main__":
    unittest.main()
