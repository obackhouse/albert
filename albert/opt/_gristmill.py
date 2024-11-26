"""Interface to `gristmill`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from albert import _default_sizes
from albert.algebra import _compose_mul
from albert.index import Index
from albert.scalar import Scalar
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Literal, Optional

    from albert.base import Base
    from albert.symmetry import Symmetry

# Try to find a pyspark context:
try:
    try:
        import collections
        import collections.abc

        collections.Iterable = collections.abc.Iterable  # type: ignore

        import dummy_spark as pyspark  # type: ignore
        from dummy_spark import SparkConf, SparkContext
    except ImportError:
        import pyspark
        from pyspark import SparkConf, SparkContext

    SPARK_CONF = SparkConf().setAppName("albert").setMaster("local")
    SPARK_CONTEXT = SparkContext(conf=SPARK_CONF)
    SPARK_CONTEXT.setLogLevel("ERROR")
except ImportError:
    pyspark = None
    SPARK_CONF = None
    SPARK_CONTEXT = None


def optimise_gristmill(
    outputs: list[Tensor],
    exprs: list[Base],
    sizes: Optional[dict[str | None, float]] = None,
    strategy: Literal["exhaust", "opt", "trav", "greedy"] = "exhaust",
    transposes: Literal["skip", "natural", "ignore"] = "natural",
    greedy_cutoff: int = -1,
    drop_cutoff: int = -1,
    **gristmill_kwargs: Any,
) -> list[tuple[Tensor, Base]]:
    """Perform common subexpression elimination on the given expression using `gristmill`.

    Args:
        outputs: The output tensors for each expression.
        exprs: The expressions to be optimised.
        sizes: The sizes of the indices.
        strategy: The optimisation strategy to use.
        transpose: The handling of transposed intermediate terms.
        greedy_cutoff: The depth cutoff for the greedy strategy. Negative values mean full
            Bron-Kerbosch backtracking.
        drop_cutoff: The depth cutoff for picking a saving in the greedy strategy. Negative
            values delegate to `greedy_cutoff`. Gives a better acceleration than `greedy_cutoff`,
            a value of `2` is recommended for very large expressions.

    Returns:
        The optimised expressions, as tuples of the output tensor and the expression.
    """
    import drudge
    import gristmill
    import sympy

    # Get the sizes
    if sizes is None:
        sizes = _default_sizes

    # Get the drudge
    dr = drudge.Drudge(SPARK_CONTEXT)

    # Find all the indices in the expressions
    indices: set[Index] = set()
    for expr in exprs:
        for node in expr.search_leaves(Tensor):
            indices.update(node.external_indices)

    # Set the indices
    substs: dict[sympy.Symbol, float] = {}
    ranges: dict[str, drudge.Range] = {}
    index_reference: dict[sympy.Symbol, tuple[str, str | None, str | None]] = {}
    for space, spin in set((i.space, i.spin) for i in indices):
        # Prepare the sympy and drudge objects
        spc = sympy.Symbol(f"N{space}{spin}", integer=True)
        rng = drudge.Range(f"{space}{spin}", 0, sizes[space])
        inds_orig = sorted([i for i in indices if i.space == space and i.spin == spin])
        inds = [i.as_sympy() for i in inds_orig]
        inds += [Index(f"idx{i}", space=space, spin=spin).as_sympy() for i in range(10)]

        # Set the dummy indices
        dr.set_dumms(rng, inds)

        # Set the substitutions
        substs[spc] = sizes[space]
        ranges[f"{space}{spin}"] = rng
        for index, index_sympy in zip(inds_orig, inds):
            index_reference[index_sympy] = (index.name, index.space, index.spin)

    # Add the resolver
    dr.add_resolver_for_dumms()

    # Get the drudge Einstein summations
    terms = []
    done: set[sympy.Symbol] = set()
    classes: dict[sympy.Symbol, type[Tensor]] = {}
    symmetries: dict[sympy.Symbol, Optional[Symmetry]] = {}
    for output, expr in zip(outputs, exprs):
        # Convert the expression to sympy
        output_sympy = output.as_sympy()
        if output.rank == 0:
            output_base = output_sympy
            output_indices = []
        else:
            output_base = output_sympy.base
            output_indices = output_sympy.indices
        expr_sympy = expr.expand().as_sympy()

        # Get the ranges on the LHS
        ranges_lhs = [
            (i, ranges[f"{index_reference[i][1]}{index_reference[i][2]}"]) for i in output_indices
        ]

        # Get the Einstein summation
        rhs = dr.einst(expr_sympy)
        rhs = rhs.simplify()
        terms.append(dr.define(output_base, *ranges_lhs, rhs))

        # Record the permutations and symbols
        tensors = [output] + list(expr.search_leaves(type_filter=Tensor))
        done = set()
        for tensor in tensors:
            base = tensor.as_sympy().base if tensor.rank else tensor.as_sympy()

            # Set the symmetry
            if tensor.symmetry is not None and base not in done:
                assert all(len(p.permutation) == tensor.rank for p in tensor.symmetry.permutations)
                perms = [
                    drudge.Perm(list(p.permutation), drudge.NEG if p.sign == -1 else drudge.IDENT)
                    for p in tensor.symmetry.permutations
                    if p.permutation != tuple(range(len(p.permutation)))
                ]
                perms = perms or [None]
                dr.set_symm(base, *perms)

            # Set the class
            if base not in classes:
                classes[base] = tensor.__class__

            # Set the symmetry
            if base not in symmetries:
                symmetries[base] = tensor.symmetry

            done.add(base)

    # Get the optimised expressions
    terms = gristmill.optimize(
        terms,
        substs=substs,
        contr_strat=getattr(gristmill.ContrStrat, strategy.upper()),
        repeated_terms_strat=getattr(gristmill.RepeatedTermsStrat, transposes.upper()),
        interm_fmt="tmp{}",
        greedy_cutoff=greedy_cutoff,
        drop_cutoff=drop_cutoff,
        **gristmill_kwargs,
    )

    # Convert the terms back to expressions
    outputs: list[Tensor] = []
    exprs: list[Base] = []
    for term in terms:
        # Convert the LHS
        base = term.lhs if isinstance(term.lhs, sympy.Symbol) else term.lhs.base
        cls = classes.get(base, Tensor)
        inds = [
            Index(index_reference[i][0], space=index_reference[i][1], spin=index_reference[i][2])
            for i in ([] if isinstance(term.lhs, sympy.Symbol) else term.lhs.indices)
        ]
        outputs.append(cls(*inds, name=base.name, symmetry=symmetries.get(base)))

        # Convert the RHS
        expr = Scalar(0.0)
        for amp in [t.amp for t in term.rhs_terms]:
            factor = Scalar(float(sympy.prod(amp.atoms(sympy.Number))))
            args: list[Tensor] = []
            for i, atom in enumerate(amp.atoms(sympy.Indexed)):
                base = atom.base
                cls = classes.get(base, Tensor)
                inds = [
                    Index(
                        index_reference[i][0],
                        space=index_reference[i][1],
                        spin=index_reference[i][2],
                    )
                    for i in atom.indices
                ]
                args.append(cls(*inds, name=base.name, symmetry=symmetries.get(base)))
            expr += _compose_mul(factor, *args)
        exprs.append(expr)

    return list(zip(outputs, exprs))
