"""Example of generating RCCSD code using `albert` and `pdaggerq`."""

import sys
import warnings

from pdaggerq import pq_helper

from albert.code.einsum import EinsumCodeGenerator
from albert.expression import Expression
from albert.opt._gristmill import optimise_gristmill
from albert.qc._pdaggerq import import_from_pdaggerq, remove_reference_energy
from albert.qc.spin import ghf_to_rhf
from albert.tensor import Tensor

# Suppress warnings since we're outputting the code to stdout
warnings.filterwarnings("ignore")

# Get the pq_helper
pq = pq_helper("fermi")

# Get the code generator
codegen = EinsumCodeGenerator(stdout=sys.stdout)
codegen.preamble()

# Find the energy expression
pq.clear()
pq.set_left_operators([["1"]])
pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
pq.simplify()
expr = pq.strings()
expr = remove_reference_energy(expr)
expr = import_from_pdaggerq(expr)
expr = ghf_to_rhf(expr).collect()
output = Tensor(name="e_cc")

# Optimise the energy expression
exprs = optimise_gristmill([Expression(output, expr)], strategy="exhaust")

# Generate the code for the energy expression
codegen("energy", [output], exprs)

# Find the T1 expression
pq.clear()
pq.set_left_operators([["e1(i,a)"]])
pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
pq.simplify()
expr_t1 = pq.strings()
expr_t1 = import_from_pdaggerq(expr_t1)
expr_t1 = ghf_to_rhf(expr_t1).collect()
output_t1 = Tensor(*expr_t1.external_indices, name="t1new")

# Find the T2 expression
pq.clear()
pq.set_left_operators([["e2(i,j,b,a)"]])
pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
pq.simplify()
expr_t2 = pq.strings()
expr_t2 = import_from_pdaggerq(expr_t2)
expr_t2 = ghf_to_rhf(expr_t2).collect()
output_t2 = Tensor(*expr_t2.external_indices, name="t2new")

# Optimise the T1 and T2 expressions
exprs = optimise_gristmill(
    [Expression(output_t1, expr_t1), Expression(output_t2, expr_t2)],
    strategy="trav",
)

# Generate the code for the T1 and T2 expressions
codegen(
    "update_amplitudes",
    [output_t1, output_t2],
    exprs,
    as_dict=True,
)

# Write the postamble (nothing for Python)
codegen.postamble()
