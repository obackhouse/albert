"""Common subexpression elimination.
"""

import itertools
from collections import defaultdict
from numbers import Number

from einfun.algebra import Add, Algebraic, Mul
from einfun.optim.cost import count_flops, memory_cost
from einfun.tensor import Symbol


def factorisation_candidates(expr, best=False):
    """
    Search the expression for candidates for factorisation.

    For example:

                Add                       Mul
           ┌─────┴─────┐             ┌─────┴──┐
          Mul         Mul    ──>    Add       c
        ┌──┴──┐     ┌──┴──┐       ┌──┴──┐
        a     c     b     c       a     b

    where the indices must match.

    Parameters
    ----------
    expr : Algebraic
        The expression.
    best : bool, optional
        If `True`, only the best candidate is returned for each
        applicable pattern identified in the tree. If `False`, all
        factorisation candidates are returned. Default value is `False`.

    Yields
    ------
    expr : Algebraic
        The factorised expression.
    """

    # Get the tree
    tree = expr.as_tree()

    # Breadth-first search for Mul(Add(...), ...) patterns
    for node in tree.bfs():
        if not isinstance(node.data[0], Add):
            continue

        # Count the patterns
        patterns = defaultdict(set)
        for i, child in enumerate(node.children):
            if not isinstance(child.data[0], Mul):
                continue
            for arg in child.data[0].args:
                patterns[arg].add(i)

        # Skip if no patterns
        if not patterns:
            continue

        # Sort the patterns by frequency
        # FIXME use cost?
        if not best:
            patterns = sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)
        else:
            patterns = [max(patterns.items(), key=lambda x: len(x[1]))]

        # If the best pattern is not repeated, skip
        if len(patterns[0][1]) < 2:
            continue

        # For each pattern, construct the replacement
        for pattern, indices in patterns:
            # Find the factors and the extra terms
            factors = []
            extra = []
            for i in range(len(node.children)):
                if i in indices:
                    j = node.children[i].data[0].args.index(pattern)
                    args = [arg for i, arg in enumerate(node.children[i].data[0].args) if i != j]
                    factors.append(Mul(*args))
                else:
                    extra.append(node.children[i].data[0])

            # Yield the new expression
            new_expr = Mul(pattern, Add(*factors)) + Add(*extra)
            expr = expr.replace(node.data[0], new_expr)

            yield expr


def intermediate_candidates(expr, best=False):
    """
    Search the expression for candidates for intermediate tensors.

    For example:

                Add
           ┌─────┴─────┐               Mul                Add
          Mul         Mul    ──>  ┌─────┴──┐   with x = ┌──┴──┐
        ┌──┴──┐     ┌──┴──┐       x        c            a     b
        a     c     b     c

    where the indices do not have to match.

    Parameters
    ----------
    expr : Algebraic
        The expression.
    best : bool, optional
        If `True`, only the best candidate is returned for each
        applicable pattern identified in the tree. If `False`, all
        factorisation candidates are returned. Default value is `False`.

    Yields
    ------
    expr : Algebraic
        The factorised expression.
    intermediates : list of tuple of (Tensor, Algebraic)
        The intermediate tensors and their expressions.
    """

    # Get the tree
    tree = expr.as_tree()

    # Breadth-first search for Mul(Add(...), ...) patterns
    for node in tree.bfs():
        if not isinstance(node.data[0], Add):
            continue

        # Count the patterns
        patterns = defaultdict(set)
        indexed_to_reindexed = {}
        reindexed_to_indexed = defaultdict(list)
        for i, child in enumerate(node.children):
            if not isinstance(child.data[0], Mul):
                continue
            for arg in child.data[0].args:
                # Replace with placeholder indices, because it may be
                # another Algebraic which is hard to strip the indices
                # from
                if not isinstance(arg, Number):
                    # Get the position of the dummy indices in this
                    # argument. Make a record for these for later
                    new_arg = arg.map_indices({idx: i for i, idx in enumerate(arg.indices)})
                    indexed_to_reindexed[arg] = new_arg
                    reindexed_to_indexed[new_arg].append(arg)
                    arg = new_arg
                patterns[arg].add(i)

        # Skip if no patterns
        if not patterns:
            continue

        # For indices spanned by the patterns, find the contracted
        # indices in all possible combinations of the children
        contraction_counts = defaultdict(int)
        for child in node.children:
            if not isinstance(child.data[0], Mul):
                continue
            args = [arg for arg in child.data[0].args if indexed_to_reindexed[arg] in patterns]

            # Loop over the combinations of the args
            for size in range(2, len(args) + 1):
                for arg_subset in itertools.combinations(args, size):
                    mul_subset = Mul(*arg_subset)

                    # If it's an outer product, skip  FIXME?
                    if mul_subset.disjoint:
                        continue

                    # Find the positions of the contracted indices
                    dummy_positions = tuple(
                        tuple(
                            arg.external_indices.index(idx)
                            for idx in mul_subset.dummy_indices
                            if idx in arg.external_indices
                        )
                        for arg in arg_subset
                    )

                    # Use the reindexed args and the contracted indices
                    # to know when contractions are equivalent within
                    # transposition
                    key_args = tuple(indexed_to_reindexed[arg] for arg in arg_subset)
                    contraction_counts[key_args, dummy_positions] += 1

        # Skip if no patterns
        if not contraction_counts:
            continue

        # Sort the patterns by frequency
        # FIXME use cost?
        if not best:
            (reargs, dummy_positions), count = max(
                contraction_counts.items(),
                key=lambda x: (len(x[0]), x[1]),
            )
            items = [(reargs, dummy_positions), count]
        else:
            items = sorted(
                contraction_counts.items(),
                key=lambda x: (len(x[0]), x[1]),
                reverse=True,
            )

        # If the best pattern is not repeated, skip
        if items[0][1] < 2:
            continue

        # For each pattern, construct the replacement and intermediates
        for (reargs, dummy_positions), count in items:
            # Initialise intermediate list
            intermediates = []

            # Get the indices and args
            indices = set.union(*(patterns[arg] for arg in reargs))
            args = tuple(reindexed_to_indexed[arg] for arg in reargs)

            # Get the intermediate
            external_indices = [
                [
                    [idx for i, idx in enumerate(arg[j].indices) if i not in dummies]
                    for arg, dummies in zip(args, dummy_positions)
                ]
                for j in range(len(indices))
            ]
            external_indices = [tuple(sum(indices, [])) for indices in external_indices]
            # TODO determine symmetry
            intermediate_symbol = Symbol(f"x{len(intermediates)}")
            intermediate = [intermediate_symbol[indices] for indices in external_indices]
            intermediates.append(
                (
                    intermediate[0],
                    Mul(*[arg[0] for arg in args]),
                )
            )

            # Find the factors and extra terms
            factors = []
            extras = []
            for i in range(len(node.children)):
                if i in indices:
                    mul_args = [
                        arg
                        for arg in node.children[i].data[0].args
                        if not any(arg == args2[i] for args2 in args)
                    ]
                    factors.append(Mul(*mul_args))
                else:
                    extras.append(node.children[i].data[0])

            # Create the new expression
            new_expr = Add(*[Mul(*tup) for tup in zip(intermediate, factors)], *extras)
            expr = expr.replace(node.data[0], new_expr)

            # See if we can factorise the expression
            while candidates := factorisation_candidates(expr, best=True):
                for candidate in candidates:
                    expr = candidate
                    break
                else:
                    break

            # We might now have a Mul(Add(...), ...) pattern, so we can
            # factorise into another intermediate
            if isinstance(expr, Algebraic):
                parts = expr.args if isinstance(expr, Add) else (expr,)
                add_args = []
                for part in parts:
                    mul_args = []
                    for arg in part.args:
                        if isinstance(arg, Add):
                            intermediate = Symbol(f"x{len(intermediates)}")
                            intermediate = intermediate[arg.external_indices]
                            intermediates.append((intermediate, arg))
                            mul_args.append(intermediate)
                        else:
                            mul_args.append(arg)
                    add_args.append(Mul(*mul_args))
                expr = Add(*add_args)

            yield expr, intermediates


def cse(
    *exprs,
    method=None,
    cost_fn=count_flops,
    memory_fn=memory_cost,
    sizes=None,
    memory_limit=None,
):
    """
    Perform common subexpression elimination on a series of
    expressions.

    Parameters
    ----------
    *exprs : list of tuple of (Tensor, Algebraic)
        A series of expressions to be optimized. For each element, the
        first element of the tuple is the output tensor, and the second
        element is the algebraic expression.
    method : str, optional
        The method to use for finding parenthesising candidates. The
        available methods are {`"brute"`, `"greedy"`, `"branch"`}.
        Default value is `None`, which is equivalent to `"brute"` for
        inputs with 4 or fewer terms, and `"greedy"` for inputs with
        more than 4 terms.
    cost_fn : callable, optional
        The cost function to use. Defaults value is `count_flops`.
    memory_fn : callable, optional
        The memory function to use. Default value is `memory_cost`.
    sizes : dict, optional
        A dictionary mapping indices to their sizes. If not provided,
        all indices are assumed to have size `10`. Default value is
        `None`.
    memory_limit : int, optional
        The maximum memory usage allowed, in units of the native
        floating point size of the tensors. If not provided, the memory
        usage is not limited. Default value is `None`.

    Returns
    -------
    exprs : list of tuple of (Tensor, Algebraic)
        The optimized expressions.
    """

    pass
