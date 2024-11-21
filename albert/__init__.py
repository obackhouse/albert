"""
****************************************************
albert: Toolbox for manipulating Einstein summations
****************************************************

The `albert` package is a toolbox for the manipulation, optimisation, and code generation for
collections of tensor contractions, with a focus on quantum chemistry.


Installation
------------

        git clone https://github.com/obackhouse/albert
        pip install .

"""  # noqa: D205, D212, D415

__version__ = "0.0.0"

_default_sizes: dict[str | None, float] = {
    "o": 200,
    "O": 20,
    "i": 180,
    "v": 800,
    "V": 20,
    "a": 780,
    "x": 3000,
    "d": 100000,
    None: 10,
}
