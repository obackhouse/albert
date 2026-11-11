"""Encoding."""

from __future__ import annotations

import json
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from string import ascii_letters
from typing import TYPE_CHECKING

from albert._core import Add, Expression, Mul, Scalar, Tensor

if TYPE_CHECKING:
    from typing import Any, TextIO

    from albert._core import Node
    from albert.metadata import Metadata


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
class EinsumEncoder(Encoder):
    """Encode expressions as NumPy-style einsum calls."""

    einsum_function: str = "np.einsum"
    """Name of the einsum function to call."""

    einsum_kwargs: dict[str, object] = field(default_factory=lambda: {"optimize": True})
    """Keyword arguments passed to each einsum call."""

    copy_function: str = "np.copy"
    """Name of the copy function used to protect aliased intermediate results."""

    metadata: Metadata | None = None
    """Metadata used to resolve tensor index spaces for ``tensor_format``."""

    def tensor_format(self, symbol: str, spaces: tuple[str, ...]) -> str:
        """Return the Python expression used to access one tensor array."""
        return symbol

    def _tensor_spaces(self, tensor: Tensor) -> tuple[str, ...]:
        if self.metadata is None:
            return ()
        index_spaces = {index: space for space, indices in self.metadata.indices.items() for index in indices}
        try:
            return tuple(index_spaces[index] for index in tensor.indices)
        except KeyError as error:
            raise ValueError(f"index {error.args[0]!r} does not belong to a metadata space") from error

    def _tensor_expression(self, tensor: Tensor) -> str:
        return self.tensor_format(tensor.symbol, self._tensor_spaces(tensor))

    def _terms(self, node: Node) -> list[list[Node]]:
        """Expand a node into additive terms of multiplicative factors."""
        if isinstance(node, Add):
            return [term for child in node.children for term in self._terms(child)]
        if isinstance(node, Mul):
            terms: list[list[Node]] = [[]]
            for child in node.children:
                terms = [left + right for left in terms for right in self._terms(child)]
            return terms
        return [[node]]

    @staticmethod
    def _index_labels(terms: list[list[Node]], output_indices: tuple[str, ...]) -> dict[str, str]:
        """Map index names to einsum's single-character labels."""
        indices: list[str] = []
        for term in terms:
            for factor in term:
                if isinstance(factor, Tensor):
                    for index in factor.indices:
                        if index not in indices:
                            indices.append(index)
        for index in output_indices:
            if index not in indices:
                indices.append(index)

        labels = {index: index for index in indices if len(index) == 1 and index in ascii_letters}
        used = set(labels.values())
        for index in indices:
            if index in labels:
                continue
            label = next((candidate for candidate in ascii_letters if candidate not in used), "")
            if not label:
                raise ValueError("einsum supports at most 52 distinct indices")
            labels[index] = label
            used.add(label)
        return labels

    def _term_call(
        self,
        factors: list[Node],
        output_indices: tuple[str, ...],
        labels: dict[str, str],
        *,
        is_assignment: bool,
    ) -> str:
        """Render one product term as an einsum call and scalar coefficient."""
        coefficient = 1.0
        tensors: list[Tensor] = []
        for factor in factors:
            if isinstance(factor, Scalar):
                coefficient *= factor.value
            elif isinstance(factor, Tensor):
                tensors.append(factor)
            else:
                raise TypeError(f"unsupported factor type: {type(factor).__name__}")

        if not tensors:
            return repr(coefficient)

        if len(tensors) == 1 and tuple(tensors[0].indices) == output_indices:
            call = self._tensor_expression(tensors[0])
            if coefficient == 1.0 and is_assignment:
                call = f"{self.copy_function}({call})"
            return self._apply_coefficient(call, coefficient)

        input_subscripts = ["".join(labels[index] for index in tensor.indices) for tensor in tensors]
        output_subscript = "".join(labels[index] for index in output_indices)
        subscripts = f"{','.join(input_subscripts)}->{output_subscript}"
        arguments = [f'"{subscripts}"', *(self._tensor_expression(tensor) for tensor in tensors)]
        arguments.extend(f"{key}={value!r}" for key, value in self.einsum_kwargs.items())
        call = f"{self.einsum_function}({', '.join(arguments)})"

        if coefficient == 1.0 and is_assignment and self._may_alias_output(tensors, output_indices):
            call = f"{self.copy_function}({call})"
        return self._apply_coefficient(call, coefficient)

    @staticmethod
    def _apply_coefficient(call: str, coefficient: float) -> str:
        if coefficient == 1.0:
            return call
        if coefficient == -1.0:
            return f"-{call}"
        return f"{coefficient!r} * {call}"

    @staticmethod
    def _may_alias_output(tensors: list[Tensor], output_indices: tuple[str, ...]) -> bool:
        """Whether a one-input einsum can return a view of its input."""
        if len(tensors) != 1:
            return False
        input_indices = tensors[0].indices
        unique_input_indices = set(input_indices)
        return len(output_indices) == len(unique_input_indices) and set(output_indices) == unique_input_indices

    def _format_terms(
        self,
        terms: list[list[Node]],
        output_indices: tuple[str, ...],
        *,
        assignment: bool = False,
    ) -> list[str]:
        labels = self._index_labels(terms, output_indices)
        return [
            self._term_call(
                factors,
                output_indices,
                labels,
                is_assignment=assignment and index == 0,
            )
            for index, factors in enumerate(terms)
        ]

    def format_node(self, node: Node) -> str:
        """Return the string representation of a node.

        Args:
            node: The node to format.

        Returns:
            The string representation of the node.
        """
        terms = self._format_terms(self._terms(node), tuple(node.external_indices))
        if not terms:
            return "0"
        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"
        return result

    def format_expression(self, expression: Expression) -> str:
        """Return the string representation of an expression.

        Args:
            expression: The expression to format.

        Returns:
            The string representation of the expression.
        """
        terms = self._terms(expression.rhs)
        calls = self._format_terms(terms, tuple(expression.lhs.indices), assignment=True)
        output = self._tensor_expression(expression.lhs)
        return "\n".join(f"{output} {'=' if index == 0 else '+='} {call}" for index, call in enumerate(calls))


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
