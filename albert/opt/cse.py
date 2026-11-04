"""Common subexpression elimination."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from functools import lru_cache
from math import isclose, prod
from typing import TYPE_CHECKING

import networkx as nx
from frozendict import frozendict

from albert import _default_sizes
from albert.algebra import Add, Mul
from albert.canon import canonicalise_exhaustive, canonicalise_indices
from albert.expression import Expression
from albert.opt.parenth import find_optimal_path, generate_paths
from albert.opt.tools import count_flops
from albert.scalar import Scalar
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Generator

    from albert.base import Base
    from albert.index import Index

    Path = tuple[tuple[int, int], ...]
    Memo = dict[tuple[tuple[Index, ...], Base], Tensor]
    Intermediates = dict[Tensor, Expression]


@dataclass(frozen=True, slots=True)
class Vertex:
    """A vertex (subtree) of a biclique.

    Attributes:
        tensors: The tensors in this vertex.
    """

    tensors: tuple[Tensor, ...]

    def indices(self) -> tuple[Index, ...]:
        """Get the unique indices in the vertex.

        Returns:
            The unique indices in the vertex.
        """
        seen: set[Index] = set()
        indices: list[Index] = []
        for tensor in self.tensors:
            for index in tensor.indices:
                if index not in seen:
                    seen.add(index)
                    indices.append(index)
        return tuple(indices)

    def __lt__(self, other: Vertex) -> bool:
        """Less-than comparison for sorting vertices."""
        return self.tensors < other.tensors


@dataclass(frozen=True, slots=True)
class Subset:
    """A subset of vertices in a biclique.

    Attributes:
        vertices: The vertices in this subset.
        coefficients: The coefficients for each vertex.
        external_indices: The external indices for this subset.
    """

    vertices: frozenset[Vertex]
    coefficients: frozendict[Vertex, float]
    external_indices: tuple[Index, ...]


@dataclass(frozen=True, slots=True)
class IndexPartition:
    """A partition of indices.

    Attributes:
        external_left: The external indices of the left subset.
        external_right: The external indices of the right subset.
        internal: The internal indices shared between the two subsets.
    """

    external_left: tuple[Index, ...]
    external_right: tuple[Index, ...]
    internal: tuple[Index, ...]

    @property
    def indices_left(self) -> tuple[Index, ...]:
        """Get all indices in the left subset."""
        return tuple(sorted(set(self.external_left) | set(self.internal)))

    @property
    def indices_right(self) -> tuple[Index, ...]:
        """Get all indices in the right subset."""
        return tuple(sorted(set(self.external_right) | set(self.internal)))

    @property
    def output(self) -> tuple[Index, ...]:
        """Get the output indices of the biclique."""
        return tuple(sorted(set(self.external_left) | set(self.external_right)))

    @classmethod
    def from_vertices(cls, expr: Mul, left: Vertex, right: Vertex) -> IndexPartition:
        """Partition indices of two vertices in a multiplication expression.

        Args:
            expr: The multiplication expression.
            left: The left vertex.
            right: The right vertex.

        Returns:
            The index partition.
        """
        external = set(expr.external_indices)
        indices_left = set(left.indices())
        indices_right = set(right.indices())
        internal = tuple(sorted(indices_left & indices_right))
        external_left = tuple(sorted(indices_left & external))
        external_right = tuple(sorted(indices_right & external))
        return cls(external_left, external_right, internal)


@dataclass(frozen=True, slots=True)
class ConstrictableBiclique:
    """A biclique that can be constricted.

    Attributes:
        left: The left subset of the biclique.
        right: The right subset of the biclique.
        coefficient: The coefficient for the biclique.
        internal_indices: The internal indices (contracted at the final step).
    """

    left: Subset
    right: Subset
    coefficient: float
    internal_indices: tuple[Index, ...]
    expression_id: int = 0

    def get_index_partition(self) -> IndexPartition:
        """Get the index partition for this biclique.

        Returns:
            The index partition for this biclique.
        """
        return IndexPartition(
            external_left=self.left.external_indices,
            external_right=self.right.external_indices,
            internal=self.internal_indices,
        )

    def gross_saving(self, sizes: dict[str | None, int]) -> int:
        """Calculate the gross saving of this biclique.

        Args:
            sizes: The sizes of the spaces.

        Returns:
            The gross saving of this biclique.
        """
        l = len(self.left.vertices)
        r = len(self.right.vertices)
        el = prod(sizes[i.space] for i in self.left.external_indices)
        er = prod(sizes[i.space] for i in self.right.external_indices)
        s = prod(sizes[i.space] for i in self.internal_indices) if self.internal_indices else 1
        naive = l * r * el * er * s
        factored = l * el * s + r * er * s + el * er * s
        return max(0, naive - factored)


def final_contraction_vertices(expr: Mul, path: Path) -> tuple[Vertex | None, Vertex | None]:
    """Get the two child vertices of the final contraction in a multiplication expression.

    Args:
        expr: The multiplication expression.
        path: The contraction path.

    Returns:
        A pair of vertices representing the left and right children of the final contraction. If
        not applicable, returns ``(None, None)``.
    """
    tensors = list(expr.search(Tensor))
    num_tensors = len(tensors)
    if num_tensors < 2 or not path:
        return None, None
    groups = [{i} for i in range(num_tensors)]
    for a, b in path[:-1]:
        if a > b:
            a, b = b, a
        merged = groups[a] | groups[b]
        del groups[b]
        del groups[a]
        groups.append(merged)
    i, j = path[-1]
    if i > j:
        i, j = j, i
    left = Vertex(tuple(tensors[k] for k in sorted(groups[i])))
    right = Vertex(tuple(tensors[k] for k in sorted(groups[j])))
    return left, right


def build_bipartite_subgraphs(
    *exprs: Expression,
    sizes: dict[str | None, int] | None = None,
    max_samples: int = 8,
    max_cpu_scaling: dict[tuple[str, ...], int] | None = None,
    max_ram_scaling: dict[tuple[str, ...], int] | None = None,
    **opt_kwargs: Any,
) -> dict[IndexPartition, nx.Graph]:
    """Build bipartite graphs for each partition of indices from the expressions.

    Args:
        *exprs: The expressions to build the graphs from.
        sizes: The sizes of the spaces in the expression.
        max_samples: The maximum number of samples to use when building the graphs.
        max_cpu_scaling: The maximum CPU scaling for the spaces in the expression.
        max_ram_scaling: The maximum RAM scaling for the spaces in the expression.
        **opt_kwargs: Additional keyword arguments for the optimization.

    Returns:
        A mapping from index partitions to bipartite graphs.

    Note:
        Multiple path sample is currently only supported when both ``max_cpu_scaling`` and
        ``max_ram_scaling`` are ``None``.
    """
    graphs: dict[IndexPartition, nx.Graph] = {}
    for expr in exprs:
        for mul in expr.rhs.expand().children:
            assert isinstance(mul, Mul)  # for mypy
            num_tensors = len(list(mul.search(Tensor)))
            if num_tensors < 2:
                continue
            scalar = prod([s.value for s in mul.search(Scalar)])

            # Get the contraction trees
            if max_cpu_scaling is None and max_ram_scaling is None:
                trees = list(
                    generate_paths(mul, sizes=sizes, max_samples=max_samples, **opt_kwargs)
                )
            else:
                trees = [
                    find_optimal_path(
                        mul,
                        sizes=sizes,
                        max_cpu_scaling=max_cpu_scaling,
                        max_ram_scaling=max_ram_scaling,
                        **opt_kwargs,
                    )
                ]
            paths = [tree.get_path() for tree in trees]
            costs = [tree.total_flops(log=None) for tree in trees]
            best_cost = min(costs) if costs else 0.0

            # Sample paths
            for tree, path, cost in zip(trees, paths, costs):
                excess_cost = cost - best_cost

                # Get the final contraction vertices
                left, right = final_contraction_vertices(mul, path)
                if left is None or right is None:
                    continue
                if right < left:
                    left, right = right, left

                # Get the index partition
                partition = IndexPartition.from_vertices(mul, left, right)

                # Add to the appropriate graph
                if partition not in graphs:
                    graphs[partition] = nx.Graph()
                    graphs[partition].graph["expression_graphs"] = {}
                graph = graphs[partition]
                expression_graphs = graph.graph["expression_graphs"]
                expression_graph = expression_graphs.setdefault(id(expr), nx.Graph())

                # Add the nodes
                u = (left, 0)
                v = (right, 1)
                if u not in graph:
                    graph.add_node(u, bipartite=0, vertex=left)
                if u not in expression_graph:
                    expression_graph.add_node(u, bipartite=0, vertex=left)
                if v not in graph:
                    graph.add_node(v, bipartite=1, vertex=right)
                if v not in expression_graph:
                    expression_graph.add_node(v, bipartite=1, vertex=right)

                # Add the edge
                if graph.has_edge(u, v):
                    graph[u][v]["excess"] = min(graph[u][v]["excess"], excess_cost)
                else:
                    graph.add_edge(u, v, excess=excess_cost, term=mul, coefficient=scalar)

                if expression_graph.has_edge(u, v):
                    expression_graph[u][v]["excess"] = min(
                        expression_graph[u][v]["excess"], excess_cost
                    )
                else:
                    expression_graph.add_edge(
                        u, v, excess=excess_cost, term=mul, coefficient=scalar
                    )

    return graphs


def decompose_edge_coefficients(
    graph: nx.Graph, left: list[Vertex], right: list[Vertex]
) -> tuple[float, frozendict[Vertex, float], frozendict[Vertex, float]]:
    r"""Attempt rank-1 decomposition of the coefficient matrix from the edges of a bipartite graph.

    .. math::
        \Phi = \lambda \pi \rho^\dagger \implies \phi_{lr} = \lambda \pi_l \rho_r

    for all edges :math:`(l, r)` in the graph.

    Args:
        graph: The bipartite graph.
        left: The vertices on the left side of the graph.
        right: The vertices on the right side of the graph.

    Returns:
        A tuple :math:`(\lambda, \pi, \rho)` where :math:`\lambda` is the overall coefficient,
        :math:`\pi` is the coefficients for the left vertices, and :math:`\rho` is the coefficients
        for the right vertices.
    """
    # TODO: find profitable cores for non-rank-1 cases

    # Pick anchors and initialise lambda
    for l0, r0 in itertools.product(left, right):
        lam = graph[l0][r0]["coefficient"]
        if lam != 0:
            break

    # If lambda is zero, all coefficients are zero
    if lam == 0.0:
        return 0.0, frozendict({v: 0.0 for v in left}), frozendict({v: 0.0 for v in right})

    # Solve for pi (left coefficients) and rho (right coefficients)
    pi = frozendict({**{l: graph[l][r0]["coefficient"] / lam for l in left}, l0: 1.0})
    rho = frozendict({**{r: graph[l0][r]["coefficient"] / lam for r in right}, r0: 1.0})

    # Verify the solution
    for l, r in itertools.product(left, right):
        if not isclose(graph[l][r]["coefficient"], lam * pi[l] * rho[r]):
            return 0.0, frozendict({v: 0.0 for v in left}), frozendict({v: 0.0 for v in right})

    return lam, pi, rho


def bipartite_vertex_sets(graph: nx.Graph) -> tuple[frozenset[Vertex], frozenset[Vertex]]:
    """Get the left and right vertex sets of a bipartite graph.

    Args:
        graph: The bipartite graph.

    Returns:
        A pair of lists containing the left and right vertices.
    """
    left = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 0}
    right = set(graph) - left
    return frozenset(left), frozenset(right)


def find_constrictable_bicliques(
    graph: nx.Graph, partition: IndexPartition
) -> Generator[ConstrictableBiclique, None, None]:
    """Find constrictable bicliques in a bipartite graph.

    Args:
        graph: The bipartite graph.
        partition: The index partition for the graph.

    Yields:
        Constrictable bicliques in the graph.
    """
    # TODO: prefer star seeds?

    expression_graphs = graph.graph.get("expression_graphs", {0: graph})
    for expression_id, expression_graph in expression_graphs.items():
        # Get the vertex sets for a single expression. A biclique must describe terms that
        # coexist in at least one expression; combining the edges of unrelated outputs can create
        # a factorization that is not valid for any one output.
        sets_left, sets_right = bipartite_vertex_sets(expression_graph)
        if not sets_left or not sets_right:
            continue

        # Build extended graph: all edges within each set are added to form cliques.
        graph_ext = expression_graph.copy()
        graph_ext.add_edges_from(itertools.combinations(sets_left, 2))
        graph_ext.add_edges_from(itertools.combinations(sets_right, 2))

        # Find all maximal bicliques in this expression's graph.
        for clique in nx.find_cliques(graph_ext):
            left = [n for n in clique if n in sets_left]
            right = [n for n in clique if n in sets_right]
            if not left or not right:
                continue

            # Ensure edges are unique.
            ids = [
                expression_graph[u][v]["term"]
                for u, v in itertools.product(left, right)
                if expression_graph.has_edge(u, v)
            ]
            if len(ids) != len(set(ids)):
                continue

            # Reject trivial bicliques.
            num_tensors_left = sum(len(expression_graph.nodes[u]["vertex"].tensors) for u in left)
            num_tensors_right = sum(len(expression_graph.nodes[v]["vertex"].tensors) for v in right)
            if num_tensors_left == num_tensors_right == 1:
                continue

            # Perform rank-1 decomposition of the coefficient matrix.
            coefficient, coefficient_left, coefficient_right = decompose_edge_coefficients(
                expression_graph, left, right
            )
            if coefficient == 0.0:
                continue

            yield ConstrictableBiclique(
                left=Subset(
                    vertices=frozenset(left),
                    coefficients=coefficient_left,
                    external_indices=tuple(sorted(partition.external_left)),
                ),
                right=Subset(
                    vertices=frozenset(right),
                    coefficients=coefficient_right,
                    external_indices=tuple(sorted(partition.external_right)),
                ),
                coefficient=coefficient,
                internal_indices=tuple(sorted(partition.internal)),
                expression_id=expression_id,
            )


def select_constrictable_bicliques(
    graphs: dict[IndexPartition, nx.Graph],
    sizes: dict[str | None, int],
    number: int = 1,
    filter_non_profitable: bool = False,
) -> list[ConstrictableBiclique]:
    """Select the most profitable constrictable bicliques from a set of bipartite graphs.

    Args:
        graphs: A mapping from index partitions to bipartite graphs.
        sizes: The sizes of the spaces.
        number: The number of bicliques to select.
        filter_non_profitable: Whether to filter out non-profitable bicliques.

    Returns:
        The most profitable constrictable bicliques.
    """
    # Find the best N bicliques
    best: list[ConstrictableBiclique] = []
    best_scores: list[int] = []
    for partition, graph in graphs.items():
        for cb in find_constrictable_bicliques(graph, partition):
            expression_graph = graph.graph.get("expression_graphs", {}).get(cb.expression_id, graph)
            gross_saving = cb.gross_saving(sizes)
            excess = sum(
                expression_graph[u][v]["excess"]
                for u in cb.left.vertices
                for v in cb.right.vertices
                if expression_graph.has_edge(u, v)
            )
            score = gross_saving - excess
            if len(best) < number:
                best.append(cb)
                best_scores.append(score)
            else:
                min_index = best_scores.index(min(best_scores))
                if score > best_scores[min_index]:
                    best[min_index] = cb
                    best_scores[min_index] = score

    # Sort by score and prune non-profitable bicliques
    if filter_non_profitable:
        for i in range(len(best) - 1, -1, -1):
            if best_scores[i] <= 0:
                del best[i]
                del best_scores[i]
    best = [cb for cb, score in sorted(zip(best, best_scores), key=lambda x: -x[1])]

    return best


def build_intermediates(
    graph: nx.Graph,
    biclique: ConstrictableBiclique,
    tensors: tuple[Tensor, Tensor],
) -> tuple[Expression, Expression]:
    """Build intermediate expressions from the given constrictable biclique.

    Args:
        graph: The bipartite graph for the biclique.
        biclique: The constrictable biclique.
        tensors: The output tensors for the left and right intermediates.

    Returns:
        A tuple of the left and right intermediate expressions.
    """

    def _build(side: Subset, tensor: Tensor) -> Expression:
        """Build an intermediate expression for one side of the biclique."""
        expr = Add.factory(
            *[
                Mul.factory(
                    Scalar.factory(float(side.coefficients[key])),
                    *graph.nodes[key]["vertex"].tensors,
                )
                for key in side.vertices
            ]
        )
        # Keep the intermediate's tensor axes in the same order as the expression that computes
        # them. The consumers will then use this exact axis order when contracting the tensor.
        return Expression(tensor.copy(*expr.external_indices), expr)

    return _build(biclique.left, tensors[0]), _build(biclique.right, tensors[1])


def rewrite_with_constriction(
    expr: Expression,
    graph: nx.Graph,
    biclique: ConstrictableBiclique,
    term: Base,
) -> Expression:
    """Rewrite the expression by replacing terms covered by the biclique.

    Args:
        expr: The original expression.
        graph: The bipartite graph for the biclique.
        biclique: The constrictable biclique.
        term: The term in the expression to be replaced.

    Returns:
        The rewritten expression.
    """
    # Find terms already covered by the biclique
    covered = {
        graph[u][v]["term"]
        for u, v in itertools.product(biclique.left.vertices, biclique.right.vertices)
        if graph.has_edge(u, v)
    }

    expression_terms = set(expr.rhs.expand().children)
    if not covered.issubset(expression_terms):
        return expr

    # Rewrite the expression
    terms: list[Base] = []
    complete = True
    for mul in expr.rhs.expand().search(Mul):
        if mul not in covered:
            terms.append(mul)
        else:
            complete = False
    if complete:
        return expr
    terms.append(term)

    return Expression(expr.lhs, Add.factory(*terms))


def rename_tensors(expr: Expression, old: str, new: str) -> Expression:
    """Rename tensors in an expression.

    Args:
        expr: The expression.
        old: The old tensor name.
        new: The new tensor name.

    Returns:
        The expression with tensors renamed.
    """

    def _rename(tensor: Tensor) -> Tensor:
        if tensor.name == old:
            return tensor.copy(name=new)
        return tensor

    return Expression(_rename(expr.lhs), expr.rhs.apply(_rename, Tensor))


def remove_trivial(exprs: list[Expression]) -> list[Expression]:
    """Collapse trivial expressions.

    Args:
        exprs: The list of expressions.

    Returns:
        The list of non-trivial expressions.
    """
    remove: set[int] = set()
    for i in range(len(exprs) - 1, -1, -1):
        rhs = exprs[i].rhs.expand()
        tensors = list(rhs.search(Tensor))
        scalar = prod([scalar.value for scalar in rhs.search(Scalar)])
        if (
            len(tensors) == 1
            and isclose(scalar, 1.0)
            and tensors[0].indices == exprs[i].lhs.indices
        ):
            name_old = exprs[i].lhs.name
            name_new = tensors[0].name
            exprs = [rename_tensors(expr, name_old, name_new) for expr in exprs]
            remove.add(i)
    return [expr for i, expr in enumerate(exprs) if i not in remove]


def renumber_intermediates(
    exprs: list[Expression], format_str: str = "__im{}__"
) -> list[Expression]:
    """Renumber intermediate tensors in a list of expressions.

    Args:
        exprs: The list of expressions.
        format_str: The format string for naming intermediate tensors.

    Returns:
        The list of expressions with renumbered intermediates.
    """
    # Get all tensor names
    tensors: set[str] = set()
    for expr in exprs:
        for tensor in expr.rhs.search(Tensor):
            tensors.add(tensor.name)
        tensors.add(expr.lhs.name)

    # Filter for intermediate tensors
    pattern = re.compile(re.escape(format_str).replace(r"\{\}", r"(\d+)"))
    tensors = {t for t in tensors if pattern.fullmatch(t)}
    tensors_list = sorted(
        tensors,
        key=lambda tensor: int(pattern.fullmatch(tensor).group(1)),  # type: ignore[union-attr]
    )

    # Build mapping
    mapping: dict[str, str] = {name: format_str.format(i) for i, name in enumerate(tensors_list)}

    # Rename tensors
    for old, new in mapping.items():
        exprs = [rename_tensors(expr, old, new) for expr in exprs]

    return exprs


def _canonicalise_expression(expr: Expression, indices: list[Index]) -> Expression:
    """Canonicalise an expression."""
    rhs_canon = canonicalise_indices(expr.rhs.canonicalise(), extra_indices=indices)
    index_map = dict(zip(expr.rhs.external_indices, rhs_canon.external_indices))
    lhs_canon = expr.lhs.map_indices(index_map)
    return Expression(lhs_canon, rhs_canon.collect())


@lru_cache(maxsize=2**14)
def _count_flops_expression(expr: Expression) -> int:
    """Count the number of floating point operations in an expression."""
    return sum(count_flops(mul) for mul in expr.rhs.expand().children)


def _check_memo(
    intermediates: Intermediates,
    intermediate: Tensor,
    memo: Memo,
    indices: list[Index],
) -> Tensor:
    """Check the memo to avoid duplicate intermediates."""
    expression = intermediates[intermediate]
    canonical = _canonicalise_expression(expression, indices)
    key = (canonical.lhs.indices, canonical.rhs)
    if key in memo:
        memo_mapped = memo[key].copy(*expression.lhs.indices)
        intermediates[intermediate] = Expression(expression.lhs, memo_mapped)
        return memo_mapped
    memo[key] = expression.lhs
    return expression.lhs


def optimise(
    expressions: list[Expression],
    sizes: dict[str | None, int] | None = None,
    intermediate_format: str = "__im{}__",
    max_path_samples: int = 16,
    max_bicliques: int = 4,
    max_cpu_scaling: dict[tuple[str, ...], int] | None = None,
    max_ram_scaling: dict[tuple[str, ...], int] | None = None,
) -> list[Expression]:
    """Optimise an expression by identifying and factoring out common subexpressions.

    Args:
        expressions: The expressions to optimise.
        sizes: The sizes of the spaces in the expression.
        intermediate_format: The format string for naming intermediate tensors.
        max_path_samples: The maximum number of samples to use when building the graphs.
        max_bicliques: The maximum number of bicliques to check.
        max_cpu_scaling: The maximum CPU scaling for the spaces in the expression.
        max_ram_scaling: The maximum RAM scaling for the spaces in the expression.

    Returns:
        A list of expressions including the original expression rewritten in terms of
        intermediates.

    Note:
        Multiple path sampling is currently only supported when both ``max_cpu_scaling`` and
        ``max_ram_scaling`` are ``None``.
    """
    if sizes is None:
        sizes = _default_sizes
    assert sizes is not None  # for mypy

    # Get all the indices
    indices: list[Index] = []
    seen: set[Index] = set()
    for expression in expressions:
        for tensor in expression.rhs.search(Tensor):
            for index in tensor.indices:
                if index not in seen:
                    seen.add(index)
                    indices.append(index)

    def _cost(expressions: list[Expression], intermediates: Intermediates) -> int:
        cost = 0
        for expr in expressions:
            cost += _count_flops_expression(expr)
        for intermediate in intermediates.values():
            cost += _count_flops_expression(intermediate)
        return cost

    def _optimise_biclique(
        expressions: list[Expression],
        intermediates: Intermediates,
        memo: Memo,
        graphs: dict[IndexPartition, nx.Graph],
        biclique: ConstrictableBiclique,
    ) -> list[Expression]:
        # Get the index partition and graph
        partition = biclique.get_index_partition()
        graph = (
            graphs[partition]
            .graph.get("expression_graphs", {})
            .get(biclique.expression_id, graphs[partition])
        )

        # Initialise intermediate tensors
        n = len(intermediates)
        left = Tensor.factory(*partition.indices_left, name=intermediate_format.format(n))
        right = Tensor.factory(*partition.indices_right, name=intermediate_format.format(n + 1))

        # Build the intermediate expressions
        intermediates[left], intermediates[right] = build_intermediates(
            graph, biclique, (left, right)
        )

        # Recursively optimise the intermediates
        intermediates[left], intermediates[right] = _optimise(
            [intermediates[left], intermediates[right]], intermediates, memo
        )

        # Check the memo to avoid duplicates
        left = _check_memo(intermediates, left, memo, indices)
        right = _check_memo(intermediates, right, memo, indices)

        # Rewrite the expression
        term = Mul.factory(Scalar.factory(float(biclique.coefficient)), left, right)
        term = canonicalise_exhaustive(term)
        expressions = [rewrite_with_constriction(e, graph, biclique, term) for e in expressions]
        for tensor in intermediates:
            intermediates[tensor] = rewrite_with_constriction(
                intermediates[tensor], graph, biclique, term
            )

        return expressions

    def _optimise(
        expressions: list[Expression], intermediates: Intermediates, memo: Memo
    ) -> list[Expression]:
        while True:
            # Build bipartite graphs and select profitable constrictable bicliques
            graphs = build_bipartite_subgraphs(
                *expressions,
                sizes=sizes,
                max_samples=max_path_samples,
                max_cpu_scaling=max_cpu_scaling,
                max_ram_scaling=max_ram_scaling,
            )
            bicliques = select_constrictable_bicliques(graphs, sizes, number=max_bicliques)
            if not bicliques:
                break

            # Optimise for the selected bicliques
            results: list[tuple[list[Expression], Intermediates, Memo]] = []
            for biclique in bicliques[:max_bicliques]:
                intermediates_new = intermediates.copy()
                memo_new = memo.copy()
                expressions_new = _optimise_biclique(
                    expressions, intermediates_new, memo_new, graphs, biclique
                )
                results.append((expressions_new, intermediates_new.copy(), memo_new.copy()))

            # Select the best result
            expressions, intermediates_new, memo_new = min(results, key=lambda x: _cost(*x[:2]))
            intermediates.clear()
            intermediates.update(intermediates_new)
            memo.clear()
            memo.update(memo_new)

        return expressions

    # Canonicalise the terms in the expressions
    expressions = list(expressions)
    for i, expression in enumerate(expressions):
        expressions[i] = _canonicalise_expression(expression, indices)

    # Recursively optimise the expression
    intermediates: Intermediates = {}
    expressions = _optimise(expressions, intermediates, {})
    expressions = list(intermediates.values()) + [*expressions]
    expressions = remove_trivial(expressions)
    expressions = renumber_intermediates(expressions, format_str=intermediate_format)

    return expressions
