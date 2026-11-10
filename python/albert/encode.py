"""Encoding."""

from __future__ import annotations

import json
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from albert._core import Add, Expression, Mul, Scalar, Tensor

if TYPE_CHECKING:
    from typing import Any, TextIO

    from albert._core import Node


@dataclass(frozen=True)
class Encoder(ABC):
    """Base class for encoders that write to a text buffer."""

    buffer: TextIO = field(default_factory=lambda: sys.stdout)
    """The buffer to write to."""

    @abstractmethod
    def format_node(self, node: Node) -> str:
        """Return the string representation of a node.

        Args:
            node: The node to format.

        Returns:
            The string representation of the node.
        """
        pass

    @abstractmethod
    def format_expression(self, expression: Expression) -> str:
        """Return the string representation of an expression.

        Args:
            expression: The expression to format.

        Returns:
            The string representation of the expression.
        """
        pass

    def encode_node(self, node: Node, *, end: str = "\n") -> None:
        """Encode a node to the configured buffer.

        Args:
            node: The node to encode.
            end: The string to append after the encoded node.
        """
        self.buffer.write(f"{self.format_node(node)}{end}")

    def encode_expression(self, expression: Expression, *, end: str = "\n") -> None:
        """Encode an expression to the configured buffer.

        Args:
            expression: The expression to encode.
            end: The string to append after the encoded expression.
        """
        self.buffer.write(f"{self.format_expression(expression)}{end}")

    def __call__(self, node: Node | Expression) -> None:
        """Encode a node or expression to the configured buffer."""
        if isinstance(node, Expression):
            self.encode_expression(node)
        else:
            self.encode_node(node)


@dataclass(frozen=True)
class _RenderedNode:
    """Rendered subtree with its precedence and leading sign."""

    magnitude: str
    precedence: int
    negative: bool = False

    @property
    def signed_text(self) -> str:
        """Return the rendered text with a leading sign if negative."""
        return f"-{self.magnitude}" if self.negative else self.magnitude


@dataclass(frozen=True)
class ReadableEncoder(Encoder):
    """Encode readable representations."""

    bracket_style: tuple[str, str] = ("[", "]")
    """Opening and closing delimiters around tensor indices."""

    index_separator: str = ", "
    """Separator between tensor indices."""

    @staticmethod
    def _format_float(value: float) -> str:
        """Format whole-valued floats without a trailing decimal point."""
        text = repr(value)
        return text[:-2] if text.endswith(".0") else text

    def render_scalar(self, scalar: Scalar) -> _RenderedNode:
        """Render a scalar while retaining its sign for sums and products."""
        value = scalar.value
        negative = value < 0 or math.copysign(1.0, value) < 0
        magnitude = abs(value) if negative else value
        return _RenderedNode(self._format_float(magnitude), 3, negative)

    def render_tensor(self, tensor: Tensor) -> _RenderedNode:
        """Render a tensor symbol and its indices."""
        text = tensor.symbol
        if tensor.indices:
            text += f"{self.bracket_style[0]}{self.index_separator.join(tensor.indices)}{self.bracket_style[1]}"
        return _RenderedNode(text, 3)

    @staticmethod
    def _flatten(node: Node, node_type: type[Add] | type[Mul]) -> list[Node]:
        """Flatten nested additions or products in encounter order."""
        if not isinstance(node, node_type):
            return [node]
        result: list[Node] = []
        for child in node.children:
            result.extend(ReadableEncoder._flatten(child, node_type))
        return result

    def _render_add(self, add: Add) -> _RenderedNode:
        """Render an addition."""
        terms = self._flatten(add, Add)
        if not terms:
            return _RenderedNode("0", 3)

        rendered = [self._render_node(term) for term in terms]
        text = rendered[0].signed_text
        for term in rendered[1:]:
            if term.negative:
                text += f" - {term.magnitude}"
            else:
                text += f" + {term.magnitude}"
        return _RenderedNode(text, 1)

    def _render_mul(self, mul: Mul) -> _RenderedNode:
        """Render a product."""
        factors = self._flatten(mul, Mul)
        if not factors:
            return _RenderedNode("1", 3)

        coefficient = 1.0
        scalar_factors: list[_RenderedNode] = []
        other_factors: list[_RenderedNode] = []
        negative = False
        for factor in factors:
            rendered = self._render_node(factor)
            negative ^= rendered.negative
            if isinstance(factor, Scalar):
                scalar_factors.append(rendered)
                coefficient *= factor.value
            else:
                other_factors.append(rendered)

        if not other_factors:
            if not scalar_factors:
                return _RenderedNode("1", 3, negative)
            magnitude = " * ".join(factor.magnitude for factor in scalar_factors)
            return _RenderedNode(magnitude, 2, negative)

        rendered_factors: list[str] = []
        if scalar_factors and abs(coefficient) != 1:
            rendered_factors.append(self._format_float(abs(coefficient)))
        for rendered_factor in other_factors:
            text = rendered_factor.magnitude
            if rendered_factor.precedence < 2:
                text = f"({text})"
            rendered_factors.append(text)
        return _RenderedNode(" ".join(rendered_factors), 2, negative)

    def _render_node(self, node: Node) -> _RenderedNode:
        """Render a node."""
        if isinstance(node, Scalar):
            return self.render_scalar(node)
        if isinstance(node, Tensor):
            return self.render_tensor(node)
        if isinstance(node, Add):
            return self._render_add(node)
        if isinstance(node, Mul):
            return self._render_mul(node)
        raise TypeError(f"unsupported node type: {type(node).__name__}")

    def format_node(self, node: Node) -> str:
        """Return the readable representation of a node.

        Args:
            node: The node to format.

        Returns:
            The readable representation of the node.
        """
        return self._render_node(node).signed_text

    def format_expression(self, expression: Expression) -> str:
        """Return the readable representation of an expression.

        Args:
            expression: The expression to format.

        Returns:
            The readable representation of the expression.
        """
        return f"{self.format_node(expression.lhs)} = {self.format_node(expression.rhs)}"


@dataclass(frozen=True)
class LatexEncoder(Encoder):
    """Encode LaTeX math representations."""

    bracket_style: tuple[str, str] = (r"\left(", r"\right)")
    """Opening and closing delimiters around additive factors."""

    index_separator: str = ""
    """Separator between tensor indices and summation indices."""

    multiplication_sign: str = ""
    """Optional sign between factors, such as ``r"\\cdot"``."""

    tensor_style: str = "{}"
    """Format string applied to each tensor symbol."""

    def _render_tensor(self, tensor: Tensor) -> _RenderedNode:
        symbol = self.tensor_style.format(tensor.symbol)
        text = f"{symbol}_{{{self.index_separator.join(tensor.indices)}}}" if tensor.indices else symbol
        summed_indices = tuple(dict.fromkeys(index for index in tensor.indices if tensor.indices.count(index) > 1))
        if summed_indices:
            text = f"\\sum_{{{self.index_separator.join(summed_indices)}}} {text}"
        return _RenderedNode(text, 1 if summed_indices else 3)

    def _render_add(self, node: Add) -> _RenderedNode:
        if not node.children:
            return _RenderedNode("0", 3)

        rendered = [self._render_node(child) for child in node.children]
        text = rendered[0].signed_text
        for term in rendered[1:]:
            if term.negative:
                text += f" - {term.magnitude}"
            else:
                text += f" + {term.magnitude}"
        return _RenderedNode(text, 1)

    def _render_mul(self, node: Mul) -> _RenderedNode:
        if not node.children:
            return _RenderedNode("1", 3)

        rendered = [self._render_node(child) for child in node.children]
        negative = False
        for factor in rendered:
            negative ^= factor.negative

        factors: list[str] = []
        ordered_factors = sorted(
            zip(node.children, rendered, strict=True),
            key=lambda item: not isinstance(item[0], Scalar),
        )
        for child, factor in ordered_factors:
            if (
                isinstance(child, Scalar)
                and abs(child.value) == 1
                and any(not isinstance(other, Scalar) for other in node.children)
            ):
                continue
            text = factor.magnitude
            if factor.precedence < 2:
                text = f"{self.bracket_style[0]}{text}{self.bracket_style[1]}"
            factors.append(text)

        separator = f" {self.multiplication_sign} " if self.multiplication_sign else " "
        text = separator.join(factors) if factors else "1"

        occurrence_count: dict[str, int] = {}
        for child in node.children:
            for index in child.external_indices:
                occurrence_count[index] = occurrence_count.get(index, 0) + 1
        summed_indices = tuple(
            index for index in occurrence_count if occurrence_count[index] > 1 and index not in node.external_indices
        )
        if summed_indices:
            text = f"\\sum_{{{self.index_separator.join(summed_indices)}}} {text}"
        return _RenderedNode(text, 1 if summed_indices else 2, negative)

    def _render_node(self, node: Node) -> _RenderedNode:
        if isinstance(node, Scalar):
            negative = node.value < 0 or (node.value == 0 and math.copysign(1.0, node.value) < 0)
            magnitude = ReadableEncoder._format_float(abs(node.value) if negative else node.value)
            return _RenderedNode(magnitude, 3, negative)
        if isinstance(node, Tensor):
            return self._render_tensor(node)
        if isinstance(node, Add):
            return self._render_add(node)
        if isinstance(node, Mul):
            return self._render_mul(node)
        raise TypeError(f"unsupported node type: {type(node).__name__}")

    def format_node(self, node: Node) -> str:
        """Return the string representation of a node.

        Args:
            node: The node to format.

        Returns:
            The string representation of the node.
        """
        return self._render_node(node).signed_text

    def format_expression(self, expression: Expression) -> str:
        """Return the string representation of an expression.

        Args:
            expression: The expression to format.

        Returns:
            The string representation of the expression.
        """
        return f"{self.format_node(expression.lhs)} = {self.format_node(expression.rhs)}"


@dataclass(frozen=True)
class TreeEncoder(Encoder):
    """Encode nodes and expressions as connector-based trees."""

    connectors: tuple[str, str, str, str] = ("─", "│", "├", "└")
    """Characters for horizontal, vertical, branch, and final branch connectors."""

    indent: int = 4
    """Number of characters used for each indentation level."""

    def _node_label(self, node: Node) -> str:
        if isinstance(node, Scalar):
            return repr(node.value)
        if isinstance(node, Tensor):
            return ReadableEncoder().render_tensor(node).signed_text
        return type(node).__name__

    def _format_tree(self, label: str, children: tuple[Node, ...]) -> str:
        indent = max(self.indent, 2)
        horizontal, vertical, branch, final = self.connectors
        lines = [label]

        def walk(node: Node, prefix: str, is_last: bool) -> None:
            connector = (final if is_last else branch) + horizontal * (indent - 2) + " "
            lines.append(f"{prefix}{connector}{self._node_label(node)}")
            next_prefix = f"{prefix}{' ' if is_last else vertical}{' ' * (indent - 1)}"
            for index, child in enumerate(node.children):
                walk(child, next_prefix, index == len(node.children) - 1)

        for index, child in enumerate(children):
            walk(child, "", index == len(children) - 1)
        return "\n".join(lines)

    def format_node(self, node: Node) -> str:
        """Return a tree representation of a node and its descendants."""
        return self._format_tree(self._node_label(node), tuple(node.children))

    def format_expression(self, expression: Expression) -> str:
        """Return a tree representation of an expression assignment."""
        return self._format_tree("Expression", (expression.lhs, expression.rhs))


@dataclass(frozen=True)
class JsonEncoder(Encoder):
    """Encode JSON representations."""

    indent: int | None = 2
    """Indentation level."""

    sort_keys: bool = True
    """Sort keys in the output JSON."""

    ensure_ascii: bool = False
    """Ensure that the output JSON is ASCII-only."""

    @staticmethod
    def _encode_node(node: Node) -> dict[str, Any]:
        if isinstance(node, Scalar):
            return {"type": "scalar", "value": node.value}
        if isinstance(node, Tensor):
            return {"type": "tensor", "symbol": node.symbol, "indices": list(node.indices)}
        if isinstance(node, Add):
            return {"type": "add", "children": [JsonEncoder._encode_node(child) for child in node.children]}
        if isinstance(node, Mul):
            return {"type": "mul", "children": [JsonEncoder._encode_node(child) for child in node.children]}
        raise TypeError(f"unsupported node type: {type(node).__name__}")

    def _dumps(self, payload: dict[str, Any]) -> str:
        return json.dumps(
            payload,
            indent=self.indent,
            sort_keys=self.sort_keys,
            ensure_ascii=self.ensure_ascii,
            allow_nan=False,
        )

    def format_node(self, node: Node) -> str:
        """Return the string representation of a node.

        Args:
            node: The node to format.

        Returns:
            The string representation of the node.
        """
        return self._dumps(self._encode_node(node))

    def format_expression(self, expression: Expression) -> str:
        """Return the string representation of an expression.

        Args:
            expression: The expression to format.

        Returns:
            The string representation of the expression.
        """
        return self._dumps(
            {
                "type": "expression",
                "lhs": self._encode_node(expression.lhs),
                "rhs": self._encode_node(expression.rhs),
            }
        )
