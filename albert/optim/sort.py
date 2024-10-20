"""Tools for sorting optimised expressions.
"""

from collections import defaultdict

import networkx as nx

from albert.algebra import Mul, Add
from albert.tensor import Tensor

# TODO the toposort is not perfect


def _build_graph(outputs, exprs, get_name=None, exclude_names=None):
    """Build a graph of dependencies in the expressions."""

    # Get the name function
    if get_name is None:
        get_name = lambda x: x.name

    # Build the graph
    graph = defaultdict(set)
    for i, (output, expr) in enumerate(zip(outputs, exprs)):
        for mul_args in expr.nested_view():
            for arg in mul_args:
                if isinstance(arg, Tensor):
                    if exclude_names is not None and get_name(arg) in exclude_names:
                        continue
                    graph[get_name(output)].add(get_name(arg))

    return graph


def split_exprs(returns, outputs, exprs, split="all"):
    """
    Split the expressions up into single contractions.

    Parameters
    ----------
    returns : tuple
        The return tensors.
    outputs : tuple
        The output tensors.
    exprs : tuple
        The optimised expressions.
    split : str, optional
        The split type. Can be `"all"` or `"returns"`. Default value is
        `"all"`.

    Returns
    -------
    outputs : tuple
        The output tensors of the split expressions.
    exprs : tuple
        The expressions split up into single contractions.
    """

    new_outputs = []
    new_exprs = []

    for output, expr in zip(outputs, exprs):
        if split == "returns" and not any(output.name == ret.name for ret in returns):
            new_outputs.append(output)
            new_exprs.append(expr)
            continue
        for mul_args in expr.nested_view():
            new_outputs.append(output)
            new_exprs.append(Mul(*mul_args))

    return new_outputs, new_exprs


def sort_exprs(returns, outputs, exprs, get_name=None):
    """
    Sort optimised expressions to optimise intermediate tensor memory
    footprint.

    Parameters
    ----------
    returns : tuple
        The return tensors.
    outputs : tuple
        The output tensors.
    exprs : tuple
        The optimised expressions.
    get_name : callable, optional
        The function to get the name of a tensor. If `None`, use the
        `name` attribute. Default value is `None`.

    Returns
    -------
    outputs : tuple
        The output tensors in the sorted order.
    exprs : tuple
        The optimised expressions in the sorted order.
    """

    # Get the name function
    if get_name is None:
        get_name = lambda x: x.name

    # Get a dictionary of the names to outputs and expressions
    outputs, exprs = split_exprs(returns, outputs, exprs, split="all")
    names = defaultdict(list)
    for output, expr in zip(outputs, exprs):
        names[get_name(output)].append((output, expr))
    names = dict(names)

    # Build the graph
    graph = _build_graph(
        outputs,
        exprs,
        get_name=get_name,
        #exclude_names=set(get_name(ret) for ret in returns),
    )

    # Sort the names
    new_outputs = []
    new_exprs = []
    _cache = {}
    def get_deps(name):
        if name in _cache:
            return _cache[name]
        deps = set()
        for dep in graph[name]:
            if dep in graph:
                deps.add(dep)
            deps.update(get_deps(dep))
        _cache[name] = deps
        return deps

    def _add(name):
        # Find the first time it's used:
        i = None
        for i, (output, expr) in enumerate(zip(new_outputs, new_exprs)):
            for mul_args in expr.nested_view():
                for arg in mul_args:
                    if isinstance(arg, Tensor) and get_name(arg) == name:
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            i = len(new_outputs)

        # Insert the outputs and expressions
        if name in names:
            for output, expr in names[name]:
                new_outputs.insert(i, output)
                new_exprs.insert(i, expr)
                i += 1

    for group in list(nx.topological_generations(nx.DiGraph(graph))):
        for name in sorted(group, key=lambda x: len(get_deps(x))):
            _add(name)

    assert len(outputs) == len(new_outputs)
    assert len(exprs) == len(new_exprs)

    return new_outputs, new_exprs
