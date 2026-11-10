from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, Self

from numpy import complex64, complex128, float32, float64
from numpy.typing import NDArray

from albert.decode import Decoder, JsonDecoder
from albert.encode import Encoder, JsonEncoder, ReadableEncoder
from albert.metadata import Metadata
from albert.options import (
    CostModel,
    DecodingMethod,
    EncodingMethod,
    NodeKind,
    PathRequests,
    PhaseFactor,
    ScalingLimits,
    ScalingMetric,
    SearchOrder,
)

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

class Encodable(Protocol):
    def format(self, encoder: Encoder | type[Encoder] | EncodingMethod = ReadableEncoder) -> str: ...
    def encode(self, encoder: Encoder | type[Encoder] | EncodingMethod = JsonEncoder) -> None: ...
    @classmethod
    def decode(cls, value: str, decoder: Decoder | type[Decoder] | DecodingMethod = JsonDecoder) -> Self: ...
    def __repr__(self) -> str: ...

class Node(Encodable):
    kind: NodeKind
    children: list[Node]
    external_indices: list[str]
    internal_indices: list[str]

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
    def map_indices(self, mapping: dict[str, str]) -> Node: ...
    def permute_indices(self, permutation: Sequence[int] | Permutation) -> Node: ...
    def __iter__(self) -> Iterator[Node]: ...
    def __contains__(self, value: object) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
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

class Expression(Encodable):
    lhs: Tensor
    rhs: Node

    def __init__(self, lhs: Tensor, rhs: Node) -> None: ...

class Permutation:
    permutation: list[int]
    phase: PhaseFactor

    def __init__(self, permutation: Sequence[int], phase: PhaseFactor = PhaseFactor.POSITIVE) -> None: ...

def canonical_key(node: Node) -> CanonicalKey: ...
def scaling_exponent(
    node: Node | Expression,
    space: str | tuple[str, ...],
    which: ScalingMetric,
    metadata: Metadata,
) -> int: ...
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
def substitute_expressions(expressions: Sequence[Expression], metadata: Metadata) -> list[Expression]: ...
def evaluate_expression[EvaluationDType: (float32, float64, complex64, complex128)](
    expressions: Sequence[Expression],
    arrays: dict[tuple[str, tuple[str, ...]], NDArray[EvaluationDType]],
    metadata: Metadata,
) -> NDArray[EvaluationDType]: ...
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
def canonicalize[N: (Node, Expression)](node: N, metadata: Metadata | None = None) -> N: ...
def expand(node: Node) -> Node: ...
def flatten(node: Node) -> Node: ...
def collect(node: Node) -> Node: ...
def fold_scalars(node: Node) -> Node: ...
def squeeze(node: Node) -> Node: ...
def prune_scalars(node: Node) -> Node: ...
def simplify(node: Node) -> Node: ...
