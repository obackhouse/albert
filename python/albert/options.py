"""Options."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from albert.decode import Decoder
    from albert.encode import Encoder


class EncodingMethod(StrEnum):
    """Encoders."""

    JSON = "json"
    READABLE = "readable"
    LATEX = "latex"
    TREE = "tree"
    EINSUM = "einsum"

    @property
    def encoder_type(self) -> type[Encoder]:
        """Return the encoder class selected by this method."""
        from albert.encode import EinsumEncoder, JsonEncoder, LatexEncoder, ReadableEncoder, TreeEncoder  # noqa: PLC0415

        return {  # type: ignore[return-value]
            EncodingMethod.JSON: JsonEncoder,
            EncodingMethod.READABLE: ReadableEncoder,
            EncodingMethod.LATEX: LatexEncoder,
            EncodingMethod.TREE: TreeEncoder,
            EncodingMethod.EINSUM: EinsumEncoder,
        }[self]


class DecodingMethod(StrEnum):
    """Decoders."""

    JSON = "json"

    @property
    def decoder_type(self) -> type[Decoder]:
        """Return the decoder class selected by this method."""
        from albert.decode import JsonDecoder  # noqa: PLC0415

        return {DecodingMethod.JSON: JsonDecoder}[self]


class ScalingMetric(StrEnum):
    """Metric for scaling exponents."""

    TIME = "time"
    MEMORY = "memory"


@dataclass(frozen=True)
class CostModel:
    """Cost model weights."""

    multiply: float = 1.0
    add: float = 1.0
    read: float = 0.1
    write: float = 0.1
    cost_factors: dict[str, int] = field(default_factory=dict)
    """Tensor cost factors keyed by tensor symbol."""
    apply_cost_factors: bool = False
    cost_factor_aggregation: Literal["product", "maximum"] = "product"

    def __post_init__(self) -> None:
        for name, weight in (
            ("multiply", self.multiply),
            ("add", self.add),
            ("read", self.read),
            ("write", self.write),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} cost weight must be finite and nonnegative")
        if not isinstance(self.apply_cost_factors, bool):
            raise ValueError("apply_cost_factors must be a bool")
        for symbol, factor in self.cost_factors.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("cost_factors keys must be non-empty tensor symbols")
            if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
                raise ValueError("cost_factors values must be positive integers")
        if self.cost_factor_aggregation not in ("product", "maximum"):
            raise ValueError("cost_factor_aggregation must be 'product' or 'maximum'")


@dataclass(frozen=True)
class ScalingLimits:
    """Scaling exponent limits."""

    time: dict[tuple[str, ...] | str, int] = field(default_factory=dict)
    memory: dict[tuple[str, ...] | str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate space keys and nonnegative exponent caps."""
        for name in ("time", "memory"):
            limits = getattr(self, name)
            for key, limit in limits.items():
                if isinstance(key, str):
                    valid_key = bool(key)
                else:
                    valid_key = (
                        isinstance(key, tuple) and bool(key) and all(isinstance(space, str) and space for space in key)
                    )
                if not valid_key:
                    raise ValueError(f"{name} keys must be a space name or nonempty tuple of names")
                if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                    raise ValueError(f"{name} limits must be nonnegative integers")


@dataclass(frozen=True)
class PathRequest:
    """Path search budget."""

    exhaustive: int = 0
    branch: int = 0
    greedy: int = 0
    dynamic: bool = False

    def __post_init__(self) -> None:
        for name in ("exhaustive", "branch", "greedy"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.dynamic, bool):
            raise ValueError("dynamic must be a bool")


@dataclass(frozen=True)
class PathRequests:
    """Path search budgets for factor number intervals."""

    bounds: tuple[int, ...]
    requests: tuple[PathRequest, ...]

    def __post_init__(self) -> None:
        if len(self.requests) != len(self.bounds) + 1:
            raise ValueError("requests must contain one more item than bounds")
        if any(isinstance(bound, bool) or not isinstance(bound, int) or bound < 1 for bound in self.bounds):
            raise ValueError("bounds must be positive integers")
        if tuple(sorted(set(self.bounds))) != self.bounds:
            raise ValueError("bounds must be strictly increasing")
        if any(not isinstance(request, PathRequest) for request in self.requests):
            raise ValueError("requests must contain PathRequest values")

    def for_size(self, operand_count: int) -> PathRequest:
        """Return the request for an operand count."""
        for bound, request in zip(self.bounds, self.requests[:-1], strict=True):
            if operand_count <= bound:
                return request
        return self.requests[-1]

    @classmethod
    def default(cls, exhaustive_limit: int = 4, max_candidates: int = 19) -> PathRequests:
        """Return the default unconstrained CSE sampling policy."""
        if exhaustive_limit < 1 or max_candidates < 0:
            raise ValueError("exhaustive_limit must be positive and max_candidates nonnegative")
        small = PathRequest(exhaustive=max(0, max_candidates - 1), dynamic=max_candidates > 0)
        return cls(
            bounds=(exhaustive_limit,),
            requests=(small, PathRequest(dynamic=True)),
        )


class PhaseFactor(IntEnum):
    """Tensor symmetry phase factors."""

    POSITIVE = 1
    NEGATIVE = -1


class NodeKind(StrEnum):
    """AST node kinds."""

    SCALAR = "scalar"
    TENSOR = "tensor"
    ADD = "add"
    MUL = "mul"


class SearchOrder(StrEnum):
    """Traversal options."""

    PREORDER = "preorder"
    POSTORDER = "postorder"
