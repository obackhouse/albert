"""Class for code generation using `einsum` in Python."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from albert.code.base import BaseCodeGenerator
from albert.misc import ExclusionSet
from albert.scalar import Scalar
from albert.tensor import Tensor

if TYPE_CHECKING:
    from typing import Optional

    from albert.base import Base
    from albert.index import Index


def _parse_indices(*index_groups: tuple[Index, ...]) -> list[tuple[int, ...]]:
    """Parse the indices in an expression for `einsum`."""
    index_numbers: dict[Index, int] = {}
    indices: list[tuple[int, ...]] = []
    for term in index_groups:
        term_indices: list[int] = []
        for index in term:
            if index not in index_numbers:
                index_numbers[index] = len(index_numbers)
            term_indices.append(index_numbers[index])
        indices.append(tuple(term_indices))
    return indices


class EinsumCodeGenerator(BaseCodeGenerator):
    """Class for code generation using `einsum` in Python."""

    _einsum_kwargs = {
        "optimize": True,
    }
    _add_spins = ExclusionSet()
    _add_spaces = ExclusionSet()

    def get_name(
        self,
        tensor: Tensor,
        add_spins: Optional[bool] = None,
        add_spaces: Optional[bool] = None,
        spin_delimiter: str = ".",
        space_delimiter: str = ".",
    ) -> str:
        """Get the name for a tensor in the code.

        Args:
            tensor: The tensor.
            add_spins: Whether to add spins to the name.
            add_spaces: Whether to add spaces to the name.
            spin_delimiter: The delimiter for spins.
            space_delimiter: The delimiter for spaces.
        """
        if tensor.name is None:
            raise ValueError("Can't generate a name for a tensor without the `name` attribute.")
        string: str = tensor.name

        # Add the spins
        _add_spins = (add_spins is None and tensor.name in self._add_spins) or add_spins
        _add_spins = _add_spins and not tensor.name.startswith("tmp")
        _add_spins = _add_spins and all(i.spin in ("a", "b") for i in tensor.external_indices)
        if _add_spins:
            spins = tuple(cast(str, i.spin) for i in tensor.external_indices)
            if len(spins):
                string += spin_delimiter + "".join(spins)

        # Add the spaces
        _add_spaces = (add_spaces is None and tensor.name in self._add_spaces) or add_spaces
        _add_spaces = _add_spaces and not tensor.name.startswith("tmp")
        _add_spaces = _add_spaces and all(i.space for i in tensor.external_indices)
        if _add_spaces:
            spaces = tuple(cast(str, i.space) for i in tensor.external_indices)
            if len(spaces):
                string += space_delimiter + "".join(spaces)

        return string

    # Module methods:

    def module_imports(self) -> None:
        """Write the module imports."""
        self.write("from types import SimpleNamespace")
        self.write("import numpy as np")

    def module_preamble(self) -> None:
        """Write the module preamble."""
        pass

    def module_docstring(self) -> None:
        """Write the module docstring."""
        metadata = self.get_metadata()
        self.write('"""Code generated by `albert` version {albert_version}.'.format(**metadata))
        self.blank()
        self.write(" * date: {date}".format(**metadata))
        self.write(" * python version: {python_version}".format(**metadata))
        self.write(" * albert version: {albert_version}".format(**metadata))
        self.write(" * caller: {caller}".format(**metadata))
        self.write(" * node: {node}".format(**metadata))
        self.write(" * system: {system}".format(**metadata))
        self.write(" * processor: {processor}".format(**metadata))
        self.write(" * release: {release}".format(**metadata))
        self.write('"""')

    def module_postamble(self) -> None:
        """Write the module postamble."""
        pass

    # Function methods:

    def function_declaration(self, name: str, args: list[Tensor]) -> None:
        """Write the function declaration.

        Args:
            name: The function name.
            args: The function arguments.
        """
        names = sorted(set(self.get_name(arg, add_spins=False, add_spaces=False) for arg in args))
        kwargs = [f"{name}=None" for name in names]
        self.write(f"def {name}({', '.join(kwargs)}, **kwargs):")

    def function_preamble(self, string: str) -> None:
        """Write the function preamble.

        Args:
            string: The function preamble.
        """
        for line in string.split("\n"):
            self.write(line)

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
        self.write(f'"""{desc}')
        self.blank()
        self.write("Args:")
        for arg, desc in args.items():
            self.write(f"    {self.get_name(arg, add_spins=False, add_spaces=False)}: {desc}")
        self.blank()
        self.write("Returns:")
        for ret, desc in rets.items():
            self.write(f"    {self.get_name(ret, add_spins=False, add_spaces=False)}: {desc}")
        self.write('"""')

    def function_postamble(self, string: str) -> None:
        """Write the function postamble.

        Args:
            string: The function postamble.
        """
        for line in string.split("\n"):
            self.write(line)

    # TODO: Remove as_dict
    def function_return(self, args: list[Tensor], as_dict: bool = False) -> None:
        """Write the function return.

        Args:
            args: The function arguments.
            as_dict: Whether to return as a dictionary. Otherwise, return as a tuple.
        """
        names = sorted(set(self.get_name(arg, add_spins=False, add_spaces=False) for arg in args))
        if as_dict:
            string = "{" + ", ".join(f'"{name}": {name}' for name in names) + "}"
        else:
            string = ", ".join(names)
        self.write(f"return {string}")

    # Expressions:

    def scalar_declaration(self, *args: Scalar) -> None:
        """Write scalar declaration(s).

        Args:
            args: The scalars.
        """
        pass  # no need to declare scalars in Python

    def tensor_declaration(self, *args: Tensor) -> None:
        """Write tensor declaration(s).

        Args:
            args: The tensors.
        """
        for arg in args:
            # If there are spins, declare the namespace. The declaration tracking in
            # `BaseCodeGenerator.__call__` doesn't suffice here because it would declare the
            # namespace for each spin case.
            spins = tuple(i.spin for i in arg.external_indices if i.spin in ("a", "b"))
            if spins:
                if not any(arg.name == name for name, _, _ in self._tensor_declared):
                    name = self.get_name(arg, add_spins=False, add_spaces=False)
                    if name.startswith("tmp"):
                        continue
                    self.write(f"{name} = SimpleNamespace()")
            else:
                pass  # no need to declare tensors in Python

    def tensor_cleanup(self, *args: Tensor) -> None:
        """Write tensor cleanup.

        Args:
            args: The tensors.
        """
        names = sorted(
            set(
                self.get_name(arg, add_spins=False, add_spaces=False)
                for arg in args
                if not any(arg.name == name for name, _, _ in self._tensor_cleaned)
            )
        )
        if names:
            self.write("del " + ", ".join(names))

    def tensor_expression(self, output: Tensor, expr: Base, declared: bool = False) -> None:
        """Write a tensor expression.

        Args:
            output: The output tensor.
            expr: The expression.
            declared: Whether the output tensor has already been declared.
        """
        expr = expr.expand()  # guarantee Add[Mul[Tensor | Scalar]]
        for i, mul in enumerate(expr._children):
            # Separate the scalar and tensors
            scalars = list(mul.search_leaves(Scalar))
            tensors = list(mul.search_leaves(Tensor))

            # Get the indices
            lhs = [tensor.external_indices for tensor in tensors]
            rhs = output.external_indices
            indices = _parse_indices(*lhs, rhs)
            assert len(indices) == len(tensors) + 1

            # Get the arguments
            args: list[str] = []
            for tensor, index in zip(tensors, indices):
                args.append(self.get_name(tensor))
                args.append(repr(index))
            args.append(repr(indices[-1]))

            # Get the operator and LHS
            operator = "=" if i == 0 and not declared else "+="
            output_name = self.get_name(output)

            # Get the factor
            factor = 1.0
            for f in scalars:
                factor *= f._value
            if abs(factor - round(factor)) < 1e-12:
                factor = int(round(factor))
            factor_string = f" * {factor}" if factor != 1 else ""

            # Write the expression
            if len(tensors) > 1:
                args_string = ", ".join(args)
                kwargs_string = ", ".join(f"{k}={v}" for k, v in self._einsum_kwargs.items())
                self.write(
                    f"{output_name} {operator} "
                    f"np.einsum({args_string}, {kwargs_string}){factor_string}"
                )
            else:
                transpose = tuple(indices[0].index(i) for i in indices[1])
                if transpose != tuple(range(len(transpose))):
                    transpose_string = f"np.transpose({args[0]}, {transpose})"
                else:
                    transpose_string = args[0]
                copy_pre = "np.copy(" if i == 0 else ""
                copy_pos = ")" if i == 0 else ""
                self.write(
                    f"{output_name} {operator} "
                    f"{copy_pre}{transpose_string}{copy_pos}{factor_string}"
                )