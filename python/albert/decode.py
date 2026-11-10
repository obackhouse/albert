"""Decoding."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from albert._core import Add, Expression, Mul, Scalar, Tensor

if TYPE_CHECKING:
    from typing import Any

    from albert._core import Node


class Decoder(ABC):
    """Base class for decoding nodes and expressions from text."""

    @abstractmethod
    def decode_node(self, value: str) -> Node:
        """Decode a node from its serialized form.

        Args:
            value: The serialized node.

        Returns:
            The decoded node.
        """
        pass

    @abstractmethod
    def decode_expression(self, value: str) -> Expression:
        """Decode an expression from its serialized form.

        Args:
            value: The serialized expression.

        Returns:
            The decoded expression.
        """
        pass

    @abstractmethod
    def __call__(self, value: str) -> Node | Expression:
        """Decode a node or expression from its serialized form."""
        pass


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _decode_node(value: Any) -> Node:
    payload = _object(value, "node")
    node_type = payload.get("type")
    if not isinstance(node_type, str):
        raise ValueError("node type must be a string")
    if node_type == "scalar":
        scalar = payload.get("value")
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)) or not math.isfinite(scalar):
            raise ValueError("scalar value must be a finite number")
        return Scalar(float(scalar))
    if node_type == "tensor":
        symbol = payload.get("symbol")
        indices = payload.get("indices")
        if not isinstance(symbol, str):
            raise ValueError("tensor symbol must be a string")
        if not isinstance(indices, list) or any(not isinstance(index, str) for index in indices):
            raise ValueError("tensor indices must be a list of strings")
        return Tensor(symbol, indices)
    if node_type in ("add", "mul"):
        children = payload.get("children")
        if not isinstance(children, list):
            raise ValueError(f"{node_type} children must be a list")
        decoded_children = [_decode_node(child) for child in children]
        return Add(decoded_children) if node_type == "add" else Mul(decoded_children)
    raise ValueError(f"unsupported node type: {node_type!r}")


class JsonDecoder(Decoder):
    """Decode JSON representations."""

    @staticmethod
    def _loads(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON: {error.msg}") from error

    def decode_node(self, value: str) -> Node:
        """Decode a node from its serialized form.

        Args:
            value: The serialized node.

        Returns:
            The decoded node.
        """
        return _decode_node(self._loads(value))

    def decode_expression(self, value: str) -> Expression:
        """Decode an expression from its serialized form.

        Args:
            value: The serialized expression.

        Returns:
            The decoded expression.
        """
        payload = _object(self._loads(value), "expression")
        if payload.get("type") != "expression":
            raise ValueError("document type must be 'expression'")
        lhs = _decode_node(payload.get("lhs"))
        if not isinstance(lhs, Tensor):
            raise ValueError("expression lhs must be a tensor")
        rhs = _decode_node(payload.get("rhs"))
        return Expression(lhs, rhs)

    def __call__(self, value: str) -> Node | Expression:
        """Decode a node or expression from its serialized form."""
        payload = _object(self._loads(value), "document")
        if payload.get("type") == "expression":
            lhs = _decode_node(payload.get("lhs"))
            if not isinstance(lhs, Tensor):
                raise ValueError("expression lhs must be a tensor")
            rhs = _decode_node(payload.get("rhs"))
            return Expression(lhs, rhs)
        return _decode_node(payload)
