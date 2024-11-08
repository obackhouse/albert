import importlib
import inspect
import os
from types import SimpleNamespace

import numpy as np

from albert.code.einsum import EinsumCodeGenerator
from albert.index import Index
from albert.opt._gristmill import optimise_gristmill
from albert.qc.ghf import Fock
from albert.qc.spin import ghf_to_uhf
from albert.tensor import Tensor


def _test_einsum(
    helper,
    returns,
    output_expr,
    kwargs,
    output_data_ref,
    codegen=EinsumCodeGenerator,
    as_dict=False,
    debug=False,
):
    caller = inspect.stack()[1].function

    with open(f"{os.path.dirname(__file__)}/_{caller}.py", "w") as file:
        codegen = codegen(stdout=file)
        codegen.preamble()
        codegen("_test_function", returns, output_expr, as_dict=as_dict)
        codegen.postamble()

    if debug:
        with open(f"{os.path.dirname(__file__)}/_{caller}.py", "r") as file:
            print(file.read())

    module = importlib.import_module(f"_{caller}")
    _test_function = module._test_function

    output_data = _test_function(**kwargs)

    try:

        def _compare(data, ref):
            if isinstance(ref, SimpleNamespace):
                for key in ref.__dict__.keys():
                    assert np.allclose(getattr(data, key), getattr(ref, key))
            else:
                assert np.allclose(data, ref)

        if isinstance(output_data, dict):
            for key in output_data:
                _compare(output_data[key], output_data_ref[key])
        else:
            _compare(output_data, output_data_ref)

    except AssertionError as e:
        raise e
    finally:
        os.remove(f"{os.path.dirname(__file__)}/_{caller}.py")


def test_einsum_code_simple(helper):
    a = Index("a")
    b = Index("b")
    c = Index("c")
    x = Tensor(a, b, name="x")
    y = Tensor(b, c, name="y")
    output = Tensor(a, c, name="output")
    expr = x * y
    output_expr = [(output, expr)]

    size = 4
    x1_data = helper.random((size, size))
    y1_data = helper.random((size, size))
    output_data_ref = (np.einsum("ab,bc->ac", x1_data, y1_data),)

    _test_einsum(
        helper,
        [output],
        output_expr,
        dict(x=x1_data, y=y1_data),
        output_data_ref,
    )


def test_einsum_code_simple_dict(helper):
    a = Index("a")
    b = Index("b")
    c = Index("c")
    x = Tensor(a, b, name="x")
    y = Tensor(b, c, name="y")
    output_1 = Tensor(a, c, name="output1")
    output_2 = Tensor(a, c, name="output2")
    expr_1 = x * y
    expr_2 = 0.5 * x * y
    output_expr = [(output_1, expr_1), (output_2, expr_2)]

    size = 4
    x1_data = helper.random((size, size))
    y1_data = helper.random((size, size))
    output_data_ref = dict(
        output1=np.einsum("ab,bc->ac", x1_data, y1_data),
        output2=0.5 * np.einsum("ab,bc->ac", x1_data, y1_data),
    )

    _test_einsum(
        helper,
        [output_1, output_2],
        output_expr,
        dict(x=x1_data, y=y1_data),
        output_data_ref,
        as_dict=True,
    )


def test_einsum_code_simple_spins_spaces(helper):
    class _EinsumCodeGenerator(EinsumCodeGenerator):
        _add_spaces = {}

    a_a = Index("a", space="o", spin="a")
    b_a = Index("b", space="o", spin="a")
    c_a = Index("c", space="o", spin="a")
    x_aa = Tensor(a_a, b_a, name="x")
    y_aa = Tensor(b_a, c_a, name="y")
    output_aa = Tensor(a_a, c_a, name="output")
    expr_aa = x_aa * y_aa
    a_b = Index("a", space="o", spin="b")
    b_b = Index("b", space="o", spin="b")
    c_b = Index("c", space="o", spin="b")
    x_bb = Tensor(a_b, b_b, name="x")
    y_bb = Tensor(b_b, c_b, name="y")
    output_bb = Tensor(a_b, c_b, name="output")
    expr_bb = 0.5 * x_bb * y_bb
    output_expr = [(output_aa, expr_aa), (output_bb, expr_bb)]

    size = 4
    x1_data = SimpleNamespace(
        aa=helper.random((size, size)),
        bb=helper.random((size, size)),
    )
    y1_data = SimpleNamespace(
        aa=helper.random((size, size)),
        bb=helper.random((size, size)),
    )
    output_data_ref = SimpleNamespace(
        aa=np.einsum("ab,bc->ac", x1_data.aa, y1_data.aa),
        bb=np.einsum("ab,bc->ac", x1_data.bb, y1_data.bb) * 0.5,
    )

    _test_einsum(
        helper,
        [output_aa, output_bb],
        output_expr,
        dict(x=x1_data, y=y1_data),
        output_data_ref,
        codegen=_EinsumCodeGenerator,
    )


def test_einsum_code_opt(helper):
    a = Index("a")
    b = Index("b")
    c = Index("c")
    d = Index("d")
    x1 = Tensor(a, b, name="x1")
    x2 = Tensor(a, b, name="x2")
    y1 = Tensor(b, c, name="y1")
    y2 = Tensor(b, c, name="y2")
    z1 = Tensor(c, d, name="z1")
    z2 = Tensor(c, d, name="z2")
    output = Tensor(a, d, name="output")
    expr = (x1 + 0.5 * x2) * (2.0 * y1 + y2) * (1.5 * z1 + -2.0 * z2)
    expr = expr.expand()
    output_expr = optimise_gristmill([output], [expr], strategy="exhaust")

    a_size = 3
    b_size = 4
    c_size = 5
    d_size = 6
    x1_data = helper.random((a_size, b_size))
    x2_data = helper.random((a_size, b_size))
    y1_data = helper.random((b_size, c_size))
    y2_data = helper.random((b_size, c_size))
    z1_data = helper.random((c_size, d_size))
    z2_data = helper.random((c_size, d_size))

    output_data_ref = np.einsum(
        "ab,bc,cd->ad",
        x1_data + 0.5 * x2_data,
        2.0 * y1_data + y2_data,
        1.5 * z1_data - 2.0 * z2_data,
    )

    _test_einsum(
        helper,
        [output],
        output_expr,
        dict(
            x1=x1_data,
            x2=x2_data,
            y1=y1_data,
            y2=y2_data,
            z1=z1_data,
            z2=z2_data,
        ),
        output_data_ref,
    )


def test_einsum_code_opt_spins(helper):
    a = Index("a", space=None)
    b = Index("b", space=None)
    c = Index("c", space=None)
    x = Fock(a, b, name="x")
    y = Fock(b, c, name="y")
    output = Fock(a, c, name="output")
    expr = x * y + 0.5 * x * y
    expr = expr.expand()
    expr = tuple(ghf_to_uhf(expr))
    output = tuple(Fock(*e.external_indices, name="output") for e in expr)
    for o, e in zip(output, expr):
        print(o, "=", e)
    output_expr = optimise_gristmill(output, expr, strategy="exhaust")

    size = 4
    x = SimpleNamespace(
        aa=helper.random((size, size)),
        bb=helper.random((size, size)),
    )
    x.aa = 0.5 * (x.aa + x.aa.T)
    x.bb = 0.5 * (x.bb + x.bb.T)
    y = SimpleNamespace(
        aa=helper.random((size, size)),
        bb=helper.random((size, size)),
    )
    y.aa = 0.5 * (y.aa + y.aa.T)
    y.bb = 0.5 * (y.bb + y.bb.T)

    output_data_ref = SimpleNamespace(
        aa=np.einsum("ab,bc->ac", x.aa, y.aa) * 1.5,
        bb=np.einsum("ab,bc->ac", x.bb, y.bb) * 1.5,
    )

    _test_einsum(
        helper,
        list(output),
        output_expr,
        dict(x=x, y=y),
        output_data_ref,
        debug=True,
    )
