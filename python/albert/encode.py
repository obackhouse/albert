"""Encoding."""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TextIO

from albert._core import Add, Expression, Mul, Node, Scalar, Tensor


@dataclass(frozen=True)
class Encoder(ABC):
    """Base class for encoders that write to a text buffer."""

    buffer: TextIO = field(default=sys.stdout)
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
    """Encode AST nodes and expressions with readable mathematical notation."""

    bracket_style: tuple[str, str] = ("(", ")")
    """Opening and closing brackets around additive factors."""

    index_separator: str = ", "
    """Separator between tensor indices."""

    def render_scalar(self, scalar: Scalar) -> _RenderedNode:
        """Render a scalar while retaining its sign for sums and products."""
        value = scalar.value
        negative = value < 0 or math.copysign(1.0, value) < 0
        return _RenderedNode(repr(abs(value) if negative else value), 3, negative)

    def render_tensor(self, tensor: Tensor) -> _RenderedNode:
        """Render a tensor symbol and its indices."""
        text = tensor.symbol
        if tensor.indices:
            text += f"[{self.index_separator.join(tensor.indices)}]"
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
            return _RenderedNode("0.0", 3)

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
            return _RenderedNode("1.0", 3)

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
                return _RenderedNode("1.0", 3, negative)
            magnitude = " * ".join(factor.magnitude for factor in scalar_factors)
            return _RenderedNode(magnitude, 2, negative)

        rendered_factors: list[str] = []
        if scalar_factors and abs(coefficient) != 1:
            rendered_factors.append(repr(abs(coefficient)))
        for rendered_factor in other_factors:
            text = rendered_factor.magnitude
            if rendered_factor.precedence < 2:
                text = f"{self.bracket_style[0]}{text}{self.bracket_style[1]}"
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
