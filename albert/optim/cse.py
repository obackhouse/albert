"""Common subexpression elimination.
"""

import itertools
from collections import defaultdict
from numbers import Number

from albert.algebra import Add, Algebraic, Mul
from albert.optim.cost import count_flops, memory_cost
from albert.tensor import Symbol


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
                    new_arg = arg.map_indices(
                        {idx: i for i, idx in enumerate(arg.external_indices)}
                    )
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
            args = [
                arg for arg in child.data[0].args
                if not isinstance(arg, Number) and indexed_to_reindexed[arg] in patterns
            ]

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
            items = [((reargs, dummy_positions), count)]
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
            print(reargs)
            print(dummy_positions)
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


def _cse_brute(*exprs, cost_fn, sizes):
    """Brute force approach to common subexpression elimination.
    """

    # Best cost and expression
    best_cost = (float("inf"), float("inf"))
    best_exprs = exprs
    best_intermediates = None

    def _iterate(*exprs, intermediates=None):
        # Loop over the expressions
        for i, expr_i in enumerate(exprs):
            # Find the factorisation candidates
            for candidate_expr, intermediates_i in intermediate_candidates(expr_i):
                # Get a list of the candidate expressions
                candidate_exprs = [None] * len(exprs)
                candidate_exprs[i] = candidate_expr

                # Replace the intermediates in the other expressions
                for j, expr_j in enumerate(exprs):
                    if i == j:
                        continue
                    for intermediate, intermediate_expr in intermediates_i:
                        expr_j = expr_j.replace(intermediate_expr, intermediate)
                    candidate_exprs[j] = expr_j

                # Check the cost
                cost = cost_fn(*candidate_exprs, sizes=sizes)
                if cost < best_cost:
                    best_cost = cost
                    best_exprs = candidate_exprs
                    best_intermediates = intermediates + intermediates_i

                # Recurse
                _iterate(*candidate_exprs, intermediates=intermediates + intermediates_i)

    # Iterate
    _iterate(*exprs, intermediates=[])

    return best_exprs, best_intermediates


def get_cost_function(cost_fn, memory_fn, sizes=None, memory_limit=None, prefer_memory=False):
    """Get the cost function.

    Parameters
    ----------
    cost_fn : callable
        The time cost function.
    memory_fn : callable
        The memory cost function.
    sizes : dict, optional
        A dictionary mapping indices to their sizes. If not provided,
        all indices are assumed to have size `10`. Default value is
        `None`.
    memory_limit : int, optional
        The maximum memory usage allowed, in units of the native
        floating point size of the tensors. If not provided, the memory
        usage is not limited. Default value is `None`.
    prefer_memory : bool, optional
        Whether to prefer parenthesising candidates with lower memory
        overhead as priority over lower FLOP cost. Default value is
        `False`.

    Returns
    -------
    cost : callable
        The cost function.
    """

    # Get the order of the cost functions and the limits
    if prefer_memory:
        cost_fns = [memory_fn, cost_fn]
        limits = [memory_limit, None]
    else:
        cost_fns = [cost_fn, memory_fn]
        limits = [None, memory_limit]

    # Get default sizes if not provided
    if sizes is None:
        sizes = defaultdict(lambda: 10)

    def _cost(*exprs):
        # Get the costs
        costs = []
        for cost_fn, limit in zip(cost_fns, limits):
            costs.append(cost_fn(*exprs, sizes=sizes))

            # Check the limit
            if limit is not None:
                if costs[-1] > limit:
                    return float("inf")

        return tuple(costs)

    return _cost


def cse(
    *exprs,
    method="brute",
    cost_fn=count_flops,
    memory_fn=memory_cost,
    sizes=None,
    memory_limit=None,
    prefer_memory=False,
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
    prefer_memory : bool, optional
        Whether to prefer parenthesising candidates with lower memory
        overhead as priority over lower FLOP cost. Default value is
        `False`.

    Returns
    -------
    exprs : tuple of Algebraic
        The optimized expressions.
    intermediates : list of tuple of (Tensor, Algebraic)
        The intermediate tensors and their expressions.
    """

    # Get the cost function
    cost = get_cost_function(
        cost_fn,
        memory_fn,
        sizes=sizes,
        memory_limit=memory_limit,
        prefer_memory=prefer_memory,
    )

    # Get the method
    if method == "brute":
        _cse = _cse_brute
    else:
        raise ValueError(f"Unknown method '{method}'.")

    # Perform the CSE
    exprs, intermediates = _cse(*exprs, cost_fn=cost, sizes=sizes)

    return exprs, intermediates
