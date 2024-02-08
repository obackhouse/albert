"""Interface to `gristmill`.
"""

try:
    from pyspark import SparkContext, SparkConf
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

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


def optimise(expr, index_groups, sizes=None):
    """
    Perform common subexpression elimination on the given expression.

    Parameters
    ----------
    expr : Base
        The expression to be optimised.
    index_groups : iterable of iterable
        Groups of equivalent indices, used to replace indices throughout
        the expression in a canonical fashion.
    sizes : dict, optional
        The sizes of the indices. If not given, the sizes are assumed to
        be equal. Default value is `None`.

    Returns
    -------
    expr : Algebraic
        The optimised expression.
    """

    if not HAS_SPARK:
        raise ValueError(f"`optimise` requires pyspark")
    if not HAS_DRUDGE:
        raise ValueError(f"`optimise` requires drudge")
    if not HAS_GRISTMILL:
        raise ValueError(f"`optimise` requires gristmill")

    # Create a Spark context
    conf = SparkConf().setAppName("albert")
    ctx = SparkContext(conf=conf)

    # Get the drudge
    dr = drudge.Drudge(ctx)

    # Set the indices
    substs = {}
    ranges = {}
    for i, group in enumerate(index_groups):
        if set(sizes[i] for i in group) != 1:
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
        inds = [sympy.Symbol(s) for s in group]
        dr.set_dumms(rng, inds)

        # Set the substitution
        substs[space] = size
        ranges[f"i{i}"] = rng

    # Add the resolver
    dr.add_resolver_for_dumms()
