from collections.abc import Callable, Sequence
from typing import Literal

from albert.metadata import Metadata
from albert.options import CostModel, NodeKind, PathRequests, PhaseFactor, ScalingLimits, SearchOrder

class CanonicalKey:
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class CostEstimate:
    multiplications: float
    additions: float
    reads: float
    writes: float
    total: float

class ParenthesizationCandidate:
    path: list[tuple[int, int]]
    cost: CostEstimate

class Node:
    kind: NodeKind
    children: list[Node]
    external_indices: list[str]
    internal_indices: list[str]

    def scaling_exponent(
        self,
        space: str | tuple[str, ...],
        which: Literal["time", "memory"],
        metadata: Metadata,
    ) -> int: ...
    def search(
        self,
        predicate: Callable[[Node], bool],
        max_depth: int | None = None,
        order: SearchOrder = SearchOrder.PREORDER,
    ) -> list[Node]: ...
    def find(
        self,
        predicate: Callable[[Node], bool],
        max_depth: int | None = None,
        order: SearchOrder = SearchOrder.PREORDER,
    ) -> Node | None: ...
    def delete(
        self,
        predicate: Callable[[Node], bool],
        max_depth: int | None = None,
    ) -> Node: ...
    def apply(
        self,
        filter: Callable[[Node], Node],
        predicate: Callable[[Node], bool] | None = None,
        max_depth: int | None = None,
    ) -> Node: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def map_indices(self, mapping: dict[str, str]) -> Node: ...
    def permute_indices(self, permutation: Sequence[int] | Permutation) -> Node: ...
    def __add__(self, other: Node | float) -> Node: ...
    def __radd__(self, other: float) -> Node: ...
    def __sub__(self, other: Node | float) -> Node: ...
    def __rsub__(self, other: float) -> Node: ...
    def __mul__(self, other: Node | float) -> Node: ...
    def __rmul__(self, other: float) -> Node: ...
    def __pos__(self) -> Node: ...
    def __neg__(self) -> Node: ...

class Scalar(Node):
    value: float

    def __init__(self, value: float) -> None: ...

class Tensor(Node):
    symbol: str
    rank: int
    indices: list[str]

    def __init__(self, symbol: str, indices: Sequence[str]) -> None: ...

class Add(Node):
    arity: int
    children: list[Node]

    def __init__(self, children: Sequence[Node]) -> None: ...

class Mul(Node):
    arity: int
    children: list[Node]

    def __init__(self, children: Sequence[Node]) -> None: ...

class Expression:
    lhs: Tensor
    rhs: Node

    def __init__(self, lhs: Tensor, rhs: Node) -> None: ...

class Permutation:
    permutation: list[int]
    phase: PhaseFactor

    def __init__(self, permutation: Sequence[int], phase: PhaseFactor = PhaseFactor.POSITIVE) -> None: ...

def canonical_key(node: Node) -> CanonicalKey: ...
def estimate_cost(node: Node, metadata: Metadata, cost_model: CostModel = ...) -> CostEstimate: ...
def parenthesization_candidates(
    node: Mul,
    metadata: Metadata,
    scaling_limits: ScalingLimits | None = ...,
    max_candidates: int = ...,
    exhaustive_limit: int = ...,
    cost_model: CostModel | None = ...,
    path_requests: PathRequests | None = ...,
) -> list[ParenthesizationCandidate]: ...
def apply_parenthesization(node: Mul, candidate: ParenthesizationCandidate) -> Node: ...
def topological_sort(expressions: Sequence[Expression]) -> list[Expression]: ...
def common_subexpression_elimination(
    expressions: Sequence[Expression],
    metadata: Metadata,
    scaling_limits: ScalingLimits | None = ...,
    cost_model: CostModel | None = ...,
    intermediate_format: str = ...,
    max_path_samples: int = ...,
    max_bicliques: int = ...,
    exhaustive_limit: int = ...,
    path_requests: PathRequests | None = ...,
) -> list[Expression]: ...
def canonicalize(node: Node, metadata: Metadata | None = None) -> Node: ...
def expand(node: Node) -> Node: ...
def flatten(node: Node) -> Node: ...
def collect(node: Node) -> Node: ...
def squeeze(node: Node) -> Node: ...
def prune(node: Node) -> Node: ...
def simplify(node: Node) -> Node: ...
