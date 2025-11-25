"""Example of generating RCCSD code using `albert` and `pdaggerq`."""

import sys
import warnings

from pdaggerq import pq_helper

from albert.code.einsum import EinsumCodeGenerator
from albert.expression import Expression
from albert.opt import optimise
from albert.qc._pdaggerq import remove_reference_energy
from albert.qc import import_expression, adapt_spin
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
expr = import_expression(expr, name="e_cc")
exprs = adapt_spin(expr, target_spin="rhf")

# Optimise the energy expression
exprs = optimise(exprs, strategy="exhaust")

# Generate the code for the energy expression
codegen("energy", [expr.lhs for expr in exprs], exprs)

# Find the T1 expression
pq.clear()
pq.set_left_operators([["e1(i,a)"]])
pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
pq.simplify()
expr_t1 = pq.strings()
expr_t1 = import_expression(expr_t1, name="t1new")
exprs_t1 = adapt_spin(expr_t1, target_spin="rhf")

# Find the T2 expression
pq.clear()
pq.set_left_operators([["e2(i,j,b,a)"]])
pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
pq.simplify()
expr_t2 = pq.strings()
expr_t2 = import_expression(expr_t2, name="t2new")
exprs_t2 = adapt_spin(expr_t2, target_spin="rhf")

# Optimise the T1 and T2 expressions
exprs = optimise(
    exprs_t1 + exprs_t2,
    strategy="trav",
)

# Generate the code for the T1 and T2 expressions
codegen(
    "update_amplitudes",
    [expr.lhs for expr in (exprs_t1 + exprs_t2)],
    exprs,
    as_dict=True,
)

# Write the postamble (nothing for Python)
codegen.postamble()
