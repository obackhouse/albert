"""Base class for code generation."""

from __future__ import annotations

import datetime
import inspect
import platform
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

from albert import __version__
from albert.opt.tools import _tensor_info, sort_expressions, split_expressions
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

    from albert.base import Base
    from albert.opt.tools import TensorInfo
    from albert.scalar import Scalar


class BaseCodeGenerator(ABC):
    """Base class for code generation."""

    _indent_size = 4

    def __init__(self, **kwargs: Any):
        """Initialise the object."""
        self._indent = 0
        self._tensor_declared: set[TensorInfo] = set()
        self._tensor_cleaned: set[TensorInfo] = set()
        self.stdout = sys.stdout
        self.__dict__.update(kwargs)

    def indent(self) -> None:
        """Indent the code."""
        self._indent += 1

    def dedent(self) -> None:
        """Dedent the code."""
        self._indent -= 1

    def reset(self) -> None:
        """Reset the caches."""
        self._tensor_declared.clear()
        self._tensor_cleaned.clear()

    @abstractmethod
    def get_name(
        self,
        tensor: Tensor,
        add_spins: Optional[bool] = None,
        add_spaces: Optional[bool] = None,
        spin_delimiter: str = "_",
        space_delimiter: str = "_",
    ) -> str:
        """Get the name for a tensor in the code.

        Args:
            tensor: The tensor.
            add_spins: Whether to add spins to the name.
            add_spaces: Whether to add spaces to the name.
            spin_delimiter: The delimiter for spins.
            space_delimiter: The delimiter for spaces.
        """
        pass

    def write(self, string: str, end: str = "\n") -> None:
        """Write a string to the output.

        Args:
            string: The string to write.
            end: The end character.
        """
        for line in string.split("\n"):
            if not len(line.strip(" ")):
                self.blank()
            else:
                self.stdout.write(" " * self._indent_size * self._indent + line + end)

    def blank(self) -> None:
        """Write a blank line."""
        self.stdout.write("\n")

    def get_metadata(self) -> dict[str, str]:
        """Get the metadata for the code generation.

        Returns:
            The metadata.
        """
        return {
            "node": platform.node(),
            "system": platform.system(),
            "processor": platform.processor(),
            "release": platform.release(),
            "user": repr(platform.uname()),
            "caller": inspect.getframeinfo(sys._getframe(1)).filename,
            "date": datetime.datetime.now().isoformat(),
            "python_version": sys.version,
            "albert_version": __version__,
        }

    def ignore_argument(self, arg: Tensor) -> bool:
        """Check if a tensor should be ignored in the function arguments.

        Args:
            arg: The tensor.

        Returns:
            Whether the tensor should be ignored.
        """
        return "tmp" in self.get_name(arg, add_spins=False, add_spaces=False)

    # Module methods:

    @abstractmethod
    def module_imports(self) -> None:
        """Write the module imports."""
        pass

    @abstractmethod
    def module_preamble(self) -> None:
        """Write the module preamble."""
        pass

    @abstractmethod
    def module_docstring(self) -> None:
        """Write the module docstring."""
        pass

    @abstractmethod
    def module_postamble(self) -> None:
        """Write the module postamble."""
        pass

    # Function methods:

    @abstractmethod
    def function_declaration(self, name: str, args: list[Tensor]) -> None:
        """Write the function declaration.

        Args:
            name: The function name.
            args: The function arguments.
        """
        pass

    @abstractmethod
    def function_preamble(self, string: str) -> None:
        """Write the function preamble.

        Args:
            string: The function preamble.
        """
        pass

    @abstractmethod
    def function_docstring(
        self,
        desc: str,
        args: dict[Tensor, str],
        rets: dict[Tensor, str],
    ) -> None:
        """Write the function docstring.

        Args:
            desc: The description of the function.
            args: The arguments and their descriptions.
            rets: The return tensors and their descriptions.
        """
        pass

    @abstractmethod
    def function_postamble(self, string: str) -> None:
        """Write the function postamble.

        Args:
            string: The function postamble.
        """
        pass

    @abstractmethod
    def function_return(self, args: list[Tensor], as_dict: bool = False) -> None:
        """Write the function return.

        Args:
            args: The function arguments.
            as_dict: Whether to return as a dictionary. Otherwise, return as a tuple.
        """
        pass

    # Expressions:

    @abstractmethod
    def scalar_declaration(self, *args: Scalar) -> None:
        """Write scalar declaration(s).

        Args:
            args: The scalars.
        """
        pass

    @abstractmethod
    def tensor_declaration(self, *args: Tensor, is_return: bool = False) -> None:
        """Write tensor declaration(s).

        Args:
            args: The tensors.
            is_return: Whether the tensor is a return tensor.
        """
        pass

    @abstractmethod
    def tensor_cleanup(self, *args: Tensor) -> None:
        """Write tensor cleanup.

        Args:
            args: The tensors.
        """
        pass

    @abstractmethod
    def tensor_expression(
        self,
        output: Tensor,
        expr: Base,
        declared: bool = False,
        is_return: bool = False,
    ) -> None:
        """Write a tensor expression.

        Args:
            output: The output tensor.
            expr: The expression.
            declared: Whether the output tensor has already been declared.
            is_return: Whether the output tensor is a return tensor.
        """
        pass

    def preamble(self) -> None:
        """Write the full module preamble."""
        self.module_docstring()
        self.blank()
        self.module_imports()
        self.blank()
        self.module_preamble()
        self.blank()

    def postamble(self) -> None:
        """Write the full module postamble."""
        self.module_postamble()

    def __call__(
        self,
        function_name: str,
        returns: list[Tensor],
        output_expr: list[tuple[Tensor, Base]],
        as_dict: bool = False,
        function_description: Optional[str] = None,
        preamble: Optional[Callable[[], None]] = None,
        postamble: Optional[Callable[[], None]] = None,
        reset_cache: bool = True,
    ) -> None:
        """Generate code for a function.

        A function takes a list of tensor contractions defined by their outputs and expressions,
        and a list of tensors that should be returned, as generates the code to arrive at the
        return tensors.

        Args:
            function_name: The name of the function.
            output_expr: The output tensors and their expressions.
            returns: The tensors to return.
            as_dict: Whether to return as a dictionary. Otherwise, return as a tuple.
            function_description: The description of the function.
            preamble: A callable to call to handle the function preamble.
            postamble: A callable to call to handle the function postamble.
            reset_cache: Whether to reset the caches before generating the code.
        """
        # Reset the caches
        if reset_cache:
            self.reset()

        # Get the arguments
        done = set()
        args = []
        for _, expr in output_expr:
            for tensor in expr.search_leaves(Tensor):
                if not self.ignore_argument(tensor) and tensor.name not in done:
                    args.append(tensor)
                    done.add(tensor.name)
        args = sorted(args, key=lambda tensor: tensor.name)

        # Get the returns
        done = set()
        rets = []
        for ret in returns:
            if ret.name not in done:
                rets.append(ret)
                done.add(ret.name)
        rets = sorted(rets, key=lambda tensor: tensor.name)

        # Write the function declaration
        self.function_declaration(function_name, args)
        self.indent()

        # Write the function docstring
        desc = function_description or f"Code generated by `albert` {__version__}."
        self.function_docstring(
            desc=desc,
            args={arg: "" for arg in args},
            rets={ret: "" for ret in rets},
        )
        self.blank()

        # Write the function preamble
        if preamble is not None:
            preamble()
            self.blank()

        # Sort the expressions
        output_expr = split_expressions(output_expr)
        output_expr = sort_expressions(output_expr)

        # Find the last appearance of each tensor
        last_appearance: dict[TensorInfo, int] = {}
        info_tensor_map: dict[TensorInfo, Tensor] = {}
        for i, (output, expr) in enumerate(output_expr):
            for tensor in expr.search_leaves(Tensor):
                info = _tensor_info(tensor)
                last_appearance[info] = i
                info_tensor_map[info] = tensor

        # Get the tensors to clean up at each step
        to_cleanup: dict[int, list[Tensor]] = defaultdict(list)
        for info, i in last_appearance.items():
            if not any(info[0] == tensor.name for tensor in args + rets):
                to_cleanup[i].append(info_tensor_map[info])

        # Write the tensor contractions
        for i, (output, expr) in enumerate(output_expr):
            # Write the tensor declaration
            info = _tensor_info(output)
            already_declared = info in self._tensor_declared
            is_return = any(info[0] == ret.name for ret in returns)
            if not already_declared:
                self.tensor_declaration(output, is_return=is_return)
                self._tensor_declared.add(info)

            # Write the tensor expression
            self.tensor_expression(output, expr, declared=already_declared, is_return=is_return)

            # Write the tensor cleanup
            already_cleaned = info in self._tensor_cleaned
            if not already_cleaned:
                _to_cleanup = to_cleanup.get(i, [])
                self.tensor_cleanup(*_to_cleanup)
                self._tensor_cleaned.update(_tensor_info(tensor) for tensor in _to_cleanup)

        # Write the function postamble
        if postamble is not None:
            self.blank()
            postamble()

        # Write the function return
        self.blank()
        self.function_return(returns, as_dict=as_dict)
        self.dedent()
        self.blank()

        # Try to flush the output
        try:
            self.stdout.flush()
        except AttributeError:
            pass
