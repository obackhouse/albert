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

from __future__ import annotations

__version__ = "0.0.0"

_default_sizes: dict[str | None, int] = {
    "o": 200,  # occupied, correlated
    "O": 20,  # occupied, active
    "i": 180,  # occupied, inactive
    "v": 800,  # virtual, correlated
    "V": 20,  # virtual, active
    "a": 780,  # virtual, inactive
    "b": 8,  # boson
    "x": 3000,  # auxiliary
    "d": 100000,  # dummy
    "k": 5,  # k-point
    None: 10,  # default
}

ALLOW_NON_EINSTEIN_NOTATION = 0
INFER_ALGEBRA_SYMMETRIES = 0
