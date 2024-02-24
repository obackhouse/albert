"""Base class for code generation.
"""

import datetime
import inspect
import platform
import sys
from collections import defaultdict

from albert import __version__
from albert.algebra import Mul
from albert.tensor import Tensor


def sort_exprs(returns, outputs, exprs):
    """
    Split the expressions into single contractions and sort them to
    minimize the number of temporary tensors.

    Parameters
    ----------
    returns : list of Tensor
        The return tensors.
    outputs : list of Tensor
        The output tensors.
    exprs : list of Algebraic
        The algebraic expressions.

    Returns
    -------
    outputs : list of Tensor
        The output tensors.
    exprs : list of Algebraic
        The algebraic expressions.
    """

    # Split the expressions up into single contractions
    tmp_outputs = []
    tmp_exprs = []
    tmp_names = []
    for output, expr in zip(outputs, exprs):
        for mul_args in expr.nested_view():
            tmp_outputs.append(output)
            tmp_exprs.append(Mul(*mul_args))
            tmp_names.append({arg.name for arg in mul_args if isinstance(arg, Tensor)})

    # Prepare a recursive function to place the expressions
    new_outputs = []
    new_exprs = []
    new_names = []
    remaining = set(range(len(tmp_outputs)))

    def _place(i):
        # Get the arguments for this index
        assert i in remaining
        output_i = tmp_outputs[i]
        expr_i = tmp_exprs[i]
        names_i = tmp_names[i]

        # Place the expression before its first use
        place = len(new_outputs)
        for j, (output_j, expr_j, names_j) in enumerate(zip(new_outputs, new_exprs, new_names)):
            if output_i.name in names_j:
                place = j
                break

        # Insert the expression
        new_outputs.insert(place, output_i)
        new_exprs.insert(place, expr_i)
        new_names.insert(place, names_i)
        remaining.remove(i)

        # Get the indices of the dependencies
        todo = []
        for j in remaining:
            if tmp_outputs[j].name in names_i:
                todo.append(j)

        # Try to place the depdendencies
        for j in reversed(sorted(todo)):
            if j in remaining:
                _place(j)

    # Place the expressions
    while remaining:
        _place(max(remaining))

    return new_outputs, new_exprs


def kernel(
    codegen,
    function_name,
    returns,
    outputs,
    exprs,
    as_dict=False,
    preamble=None,
    postamble=None,
):
    """
    Generate the code for a function using a list of expressions.

    Parameters
    ----------
    codegen : CodeGen
        The code generation object.
    function_name : str
        The name of the function.
    returns : list of Tensor
        The return tensors.
    outputs : list of Tensor
        The output tensors.
    exprs : list of Algebraic
        The algebraic expressions.
    as_dict : bool, optional
        Whether to return the outputs as a dictionary. Default value is
        `False`.
    preamble : str, optional
        Preamble to add to the function. Default value is `None`.
    postamble : str, optional
        Postamble to add to the function. Default value is `None`.
    """

    # Get the arguments
    args = sorted(
        set(
            arg.name
            for expr in exprs
            for mul_args in expr.nested_view()
            for arg in mul_args
            if isinstance(arg, Tensor) and not arg.name.startswith("tmp")
        ),
    )
    rets = sorted(set([ret.name for ret in returns]))

    # Write the function declaration
    codegen.function_declaration(function_name, args)
    codegen.indent()

    # Write the function docstring
    metadata = codegen.get_metadata()
    parameters_str = "\n".join([f"{arg} : array" for arg in args])
    returns_str = "\n".join([f"{ret} : array" for ret in rets])
    docstring = f"Code generated by albert {metadata['albert_version']} on {metadata['date']}.\n"
    docstring += "\n"
    docstring += "Parameters\n----------\n"
    docstring += parameters_str
    docstring += "\n\n"
    docstring += "Returns\n-------\n"
    docstring += returns_str
    codegen.function_docstring(docstring)
    codegen.blank()

    # Write the function preamble
    if preamble:
        codegen.function_preamble(preamble)

    # Sort the expressions
    outputs, exprs = sort_exprs(returns, outputs, exprs)

    # Find the last appearance of each tensor
    last_appearance = {}
    for i, (output, expr) in enumerate(zip(outputs, exprs)):
        for mul_args in expr.nested_view():
            for arg in mul_args:
                if isinstance(arg, Tensor) and arg.rank > 0:
                    last_appearance[arg.name] = i

    # Get the tensors to cleanup at each step
    to_cleanup = defaultdict(list)
    for name, i in last_appearance.items():
        if not any(name == r for r in rets) and not any(name == a for a in args):
            to_cleanup[i].append(name)

    # Write the function declarations
    declared = set()
    for i, (output, expr) in enumerate(zip(outputs, exprs)):
        # Write the declarations
        already_declared = codegen.get_name(output) in declared
        if not already_declared:
            if output.rank == 0:
                codegen.scalar_declaration(output)
            else:
                codegen.tensor_declaration(output)
            declared.add(codegen.get_name(output))

        # Write the expression
        codegen.algebraic_expression(output, expr, already_declared=already_declared)

        # Write the cleanup
        codegen.tensor_cleanup(*to_cleanup.get(i, []))

    # Write the function postamble
    if postamble:
        codegen.function_postamble(postamble)

    # Write the function return
    codegen.blank()
    codegen.function_return(rets, as_dict=as_dict)
    codegen.dedent()
    codegen.blank()


class CodeGen:
    """Base class for code generation."""

    def __init__(self, name_generator=None, **kwargs):
        self._indent = 0
        self._name_generator = name_generator
        self.stdout = sys.stdout
        self.__dict__.update(kwargs)

    def indent(self):
        """Indent the code."""
        self._indent += 1

    def dedent(self):
        """Dedent the code."""
        self._indent -= 1

    def get_name(self, tensor):
        """Get a name."""
        if self._name_generator is not None:
            return self._name_generator(tensor)
        return tensor.name

    def write(self, string, end="\n"):
        """Write a string."""
        for line in string.split("\n"):
            if all(x == " " for x in line):
                self.blank()
            else:
                self.stdout.write("    " * self._indent + line + end)

    def blank(self):
        """Write a blank line."""
        self.stdout.write("\n")

    def get_metadata(self):
        """Get the metadata."""
        return {
            "node": platform.node(),
            "system": platform.system(),
            "processor": platform.processor(),
            "release": platform.release(),
            "user": platform.uname(),
            "caller": inspect.getframeinfo(sys._getframe(1)).filename,
            "date": datetime.datetime.now().isoformat(),
            "python_version": sys.version,
            "albert_version": __version__,
        }

    def module_imports(self):
        """Write the module imports."""
        raise NotImplementedError

    def module_preamble(self, preamble):
        """Write the module preamble."""
        raise NotImplementedError

    def module_docstring(self):
        """Write the module docstring."""
        raise NotImplementedError

    def module_postamble(self, postamble):
        """Write the module postamble."""
        raise NotImplementedError

    def function_declaration(self, name, args):
        """Write a function declaration."""
        raise NotImplementedError

    def function_preamble(self, preamble):
        """Write the function preamble."""
        raise NotImplementedError

    def function_docstring(self, docstring):
        """Write the function docstring."""
        raise NotImplementedError

    def function_postamble(self, postamble):
        """Write the function postamble."""
        raise NotImplementedError

    def function_return(self, args, as_dict=False):
        """Write the function return."""
        raise NotImplementedError

    def scalar_declaration(self, *args):
        """Write a scalar declaration."""
        raise NotImplementedError

    def tensor_declaration(self, *args):
        """Write a tensor declaration."""
        raise NotImplementedError

    def tensor_cleanup(self, *args):
        """Write a tensor cleanup."""
        raise NotImplementedError

    def algebraic_expression(self, output, expr, already_declared=False):
        """Write an algebraic expression."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Write a function using a list of expressions."""
        return kernel(self, *args, **kwargs)

    def preamble(self, imports=None, preamble=""):
        """Write the preamble."""
        self.module_docstring()
        self.blank()
        if imports:
            self.module_imports(imports=imports)
        else:
            self.module_imports()
        self.blank()
        self.module_preamble(preamble)
        self.blank()

    def postamble(self, postamble=""):
        """Write the postamble."""
        self.module_postamble(postamble)
