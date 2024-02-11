"""Interface to `gristmill`.
"""

try:
    from pyspark import SparkConf, SparkContext

    SPARK_CONF = SparkConf().setAppName("albert")
    SPARK_CTX = SparkContext(conf=SPARK_CONF)
    SPARK_CTX.setLogLevel("ERROR")
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

try:
    import sympy

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import drudge

    HAS_DRUDGE = True
except ImportError:
    HAS_DRUDGE = False

try:
    import gristmill

    HAS_GRISTMILL = True
except ImportError:
    HAS_GRISTMILL = False

from albert.algebra import Mul
from albert.tensor import Symbol, Tensor


def optimise(
    *outputs_and_exprs, index_groups=None, sizes=None, strategy="exhaust", **gristmill_kwargs
):
    """
    Perform common subexpression elimination on the given expression.

    Parameters
    ----------
    outputs_and_exprs : list of (Tensor, Algebraic)
        The expressions to be optimised. The first element of each tuple
        is the output tensor, and the second element is the expression
        for the output tensor.
    index_groups : iterable of iterable
        Groups of equivalent indices, used to replace indices throughout
        the expression in a canonical fashion.
    sizes : dict, optional
        The sizes of the indices. If not given, the sizes are assumed to
        be equal. Default value is `None`.
    strategy : str, optional
        The optimisation strategy to use. Default value is `"exhaust"`.
    **gristmill_kwargs
        Additional keyword arguments to be passed to
        `gristmill.optimize`.

    Returns
    -------
    outputs_and_exprs_opt : list of (Tensor, Algebraic)
        The optimised expressions.
    """

    # Check the input
    if index_groups is None:
        raise ValueError("`index_groups` must be provided")

    # Check for dependencies
    if not HAS_SYMPY:
        raise ValueError("`optimise` requires sympy")
    if not HAS_SPARK:
        raise ValueError("`optimise` requires pyspark")
    if not HAS_DRUDGE:
        raise ValueError("`optimise` requires drudge")
    if not HAS_GRISTMILL:
        raise ValueError("`optimise` requires gristmill")

    # Get the drudge
    dr = drudge.Drudge(SPARK_CTX)

    # Get some convenience dictionaries
    index_to_group = {str(idx): i for i, group in enumerate(index_groups) for idx in group}

    # Set the indices
    substs = {}
    ranges = {}
    indices = {}
    for i, group in enumerate(index_groups):
        if len(set(sizes[symb] for symb in group)) != 1:
            raise ValueError(f"Indices in group {i} have different sizes")
        if not any(g in sizes for g in group):
            raise ValueError(f"Size of group {i} not given")

        # Get the size of this group
        size = None
        for g in group:
            if g in sizes:
                size = sizes[g]
                break

        # Build the index space
        space = sympy.Symbol(f"Ni{i}")
        rng = drudge.Range(f"i{i}", 0, sizes[group[0]])
        inds = [sympy.Symbol(str(s)) for s in group]
        dr.set_dumms(rng, inds)

        # Set the substitution
        substs[space] = size
        ranges[f"i{i}"] = rng

        # Record the indices
        for ind_sympy, ind in zip(inds, group):
            indices[ind_sympy] = ind

    # Add the resolver
    dr.add_resolver_for_dumms()

    # Get the drudge einstein summations
    terms = []
    done = set()
    symbols = {}
    for output, expr in outputs_and_exprs:
        # Convert the expression to sympy
        output_sympy = output.as_sympy()
        expr_sympy = expr.expand().as_sympy()

        # Get the ranges on the LHS
        ranges_lhs = [(i, ranges[f"i{index_to_group[i.name]}"]) for i in output_sympy.indices]

        # Get the einstein summation
        rhs = dr.einst(expr_sympy)
        terms.append(dr.define(output_sympy.base, *ranges_lhs, rhs))

        # Record the permutations and symbols
        tensors = [output] + sum(expr.nested_view(), [])
        tensors = [t for t in tensors if isinstance(t, Tensor)]
        done = set()
        for tensor in tensors:
            base = tensor.as_sympy().base

            # Set the symmetry
            if tensor.symmetry and base not in done:
                perms = [
                    drudge.Perm(list(p.permutation), {-1: drudge.NEG, 1: drudge.IDENT}[p.sign])
                    for p in tensor.symmetry.permutations
                    if p.permutation != tuple(range(tensor.rank))
                ]
                perms = perms or [None]
                dr.set_symm(base, *perms)

            # Set the symbol
            if base not in symbols:
                symbols[base] = tensor.as_symbol()

            done.add(base)

    # Optimise the expression
    terms = gristmill.optimize(
        terms,
        substs=substs,
        contr_strat=getattr(gristmill.ContrStrat, strategy.upper()),
        interm_fmt="tmp{}",
        **gristmill_kwargs,
    )

    # Convert the expressions back to `albert` objects
    outputs = []
    exprs = []
    for term in terms:
        # Convert the LHS
        base = term.lhs if isinstance(term.lhs, sympy.Symbol) else term.lhs.base
        base = symbols.get(base, Symbol(str(base)))  # TODO symmetry for tmp
        inds = [] if isinstance(term.lhs, sympy.Symbol) else term.lhs.indices
        inds = [indices[ind] for ind in inds]
        outputs.append(base[tuple(inds)])

        # Convert the RHS
        expr = 0
        for amp in [t.amp for t in term.rhs_terms]:
            factor = float(sympy.prod(amp.atoms(sympy.Number)))
            mul_args = [factor]
            for i, tensor in enumerate(amp.atoms(sympy.Indexed)):
                base = symbols.get(tensor.base, Symbol(str(tensor.base)))
                inds = [indices[ind] for ind in tensor.indices]
                mul_args.append(base[tuple(inds)])
            expr += Mul(*mul_args)
        exprs.append(expr)

    return list(zip(outputs, exprs))
