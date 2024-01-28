"""Base class for code generation.
"""


class CodeGen:
    """Base class for code generation.
    """

    def __init__(self, **kwargs):
        """Initialize.
        """
        self.__dict__.update(kwargs)
        self._indent = 0

    def indent(self):
        """Indent the code.
        """
        self._indent += 1

    def dedent(self):
        """Dedent the code.
        """
        self._indent -= 1

    def module_imports(self):
        """Write the module imports.
        """
        raise NotImplementedError

    def module_preamble(self):
        """Write the module preamble.
        """
        raise NotImplementedError

    def module_docstring(self):
        """Write the module docstring.
        """
        raise NotImplementedError

    def module_postamble(self):
        """Write the module postamble.
        """
        raise NotImplementedError

    def function_declaration(self):
        """Write a function declaration.
        """
        raise NotImplementedError

    def function_preamble(self):
        """Write the function preamble.
        """
        raise NotImplementedError

    def function_docstring(self):
        """Write the function docstring.
        """
        raise NotImplementedError

    def function_postamble(self):
        """Write the function postamble.
        """
        raise NotImplementedError

    def function_return(self):
        """Write the function return.
        """
        raise NotImplementedError

    def scalar_declaration(self):
        """Write a scalar declaration.
        """
        raise NotImplementedError

    def tensor_declaration(self):
        """Write a tensor declaration.
        """
        raise NotImplementedError

    def tensor_cleanup(self):
        """Write a tensor cleanup.
        """
        raise NotImplementedError

    def algebraic_expression(self):
        """Write an algebraic expression.
        """
        raise NotImplementedError
