"""Interface to `wick`.
"""

import itertools
import re
from fractions import Fraction

try:
    from qwick.convenience import commute, ep11, one_e, one_p, two_e, two_p
    from qwick.expression import Expression, Term
    from qwick.index import Idx
    from qwick.operator import BOperator, FOperator, Sigma, Tensor, TensorSym
    from qwick.wick import apply_wick  # noqa: F401
except ImportError:
    from wick.convenience import commute, ep11, one_e, one_p, two_e, two_p
    from wick.expression import Expression, Term
    from wick.index import Idx
    from wick.operator import BOperator, Sigma, Tensor, FOperator, TensorSym
    from wick.wick import apply_wick  # noqa: F401

from albert.algebra import Add, Mul
from albert.qc._pdaggerq import _is_number
from albert.qc.ghf import (
    ERI,
    L1,
    L2,
    L3,
    LS1,
    LS2,
    LU11,
    LU12,
    S1,
    S2,
    T1,
    T2,
    T3,
    U11,
    U12,
    BosonicHamiltonian,
    BosonicInteractionHamiltonian,
    Delta,
    ElectronBosonConjHamiltonian,
    ElectronBosonHamiltonian,
    Fock,
)
from albert.qc.uhf import SpinIndex
from albert.symmetry import antisymmetric_permutations


def import_from_wick(expr, index_spins=None):
    """Import terms from `wick` output into the internal format.

    Parameters
    ----------
    expr : wick.expression.AExpression
        Expression from `wick` output, corresponding to a single tensor
        output.
    index_spins : dict, optional
        Dictionary mapping indices to spins. If not provided, the
        indices are not constrained to a single spin. Default value is
        `None`.

    Returns
    -------
    expr : Algebraic
        Expression in the internal format.
    """

    # Get the index spins
    if index_spins is None:
        index_spins = {}

    contractions = []
    for term in str(expr).split("\n"):
        # Parse the term
        sign, term = term.split()
        term = term.replace("}", "} ")
        i = 0
        while i < len(term) and (term[i].isdigit() or term[i] == "."):
            i += 1
        factor = float(sign + term[:i])
        term = term[i:].split()
        term = [t for t in term if not t.startswith(r"\sum")]

        # Convert symbols
        symbols = [_convert_symbol(symbol, index_spins=index_spins) for symbol in term]

        # Add to the list of contractions
        part = Mul(*symbols) * factor
        contractions.append(part)

    # Add all contractions together
    expr = Add(*contractions)

    return expr


def _convert_symbol(symbol, index_spins=None):
    """Convert a `wick` symbol to the internal format.

    Parameters
    ----------
    symbol : str
        Symbol to convert.
    index_spins : dict, optional
        Dictionary mapping indices to spins. If not provided, the
        indices are not constrained to a single spin. Default value is
        `None`.

    Returns
    -------
    out : Tensor or Number
        Converted symbol.
    """

    # Swap the bosonic indices
    for src, dst in zip("IJKLMN", ("x", "y", "z", "b0", "b1", "b2")):
        symbol = symbol.replace(src, dst)

    if _is_number(symbol):
        # factor
        return float(symbol)

    elif re.match(r"f_\{[a-z][a-z]\}", symbol):
        # f_{ij}
        indices = tuple(symbol[3:5])
        tensor_symbol = Fock

    elif re.match(r"v_\{[a-z][a-z][a-z][a-z]\}", symbol):
        # v_{ijkl}
        indices = tuple(symbol[3:7])
        tensor_symbol = ERI

    elif re.match(r"G_\{[a-z]\}", symbol):
        # G_{I}
        indices = tuple(symbol[3:4])
        tensor_symbol = BosonicHamiltonian

    elif re.match(r"w_\{[a-z][a-z]\}", symbol):
        # w_{IJ}
        indices = tuple(symbol[3:5])
        tensor_symbol = BosonicInteractionHamiltonian

    elif re.match(r"gc_\{[a-z][a-z][a-z]\}", symbol):
        # gc_{Iij}
        indices = tuple(symbol[4:7])
        tensor_symbol = ElectronBosonConjHamiltonian

    elif re.match(r"g_\{[a-z][a-z][a-z]\}", symbol):
        # g_{Iij}
        indices = tuple(symbol[3:6])
        tensor_symbol = ElectronBosonHamiltonian

    elif re.match(r"t1_\{[a-z][a-z]\}", symbol):
        # t1_{ij}
        indices = tuple(symbol[4:6])
        tensor_symbol = T1

    elif re.match(r"t2_\{[a-z][a-z][a-z][a-z]\}", symbol):
        # t2_{ijkl}
        indices = tuple(symbol[4:8])
        tensor_symbol = T2

    elif re.match(r"t3_\{[a-z][a-z][a-z][a-z][a-z][a-z]\}", symbol):
        # t3_{ijklmn}
        indices = tuple(symbol[4:10])
        tensor_symbol = T3

    elif re.match(r"l1_\{[a-z][a-z]\}", symbol):
        # l1_{ij}
        indices = tuple(symbol[4:6])
        tensor_symbol = L1

    elif re.match(r"l2_\{[a-z][a-z][a-z][a-z]\}", symbol):
        # l2_{ijkl}
        indices = tuple(symbol[4:8])
        tensor_symbol = L2

    elif re.match(r"l3_\{[a-z][a-z][a-z][a-z][a-z][a-z]\}", symbol):
        # l3_{ijklmn}
        indices = tuple(symbol[4:10])
        tensor_symbol = L3

    elif re.match(r"ls1_\{[a-z]\}", symbol):
        # ls1_{I}
        indices = tuple(symbol[5:6])
        tensor_symbol = LS1

    elif re.match(r"ls2_\{[a-z][a-z]\}", symbol):
        # ls2_{IJ}
        indices = tuple(symbol[5:7])
        tensor_symbol = LS2

    elif re.match(r"s1_\{[a-z]\}", symbol):
        # s1_{I}
        indices = tuple(symbol[4:5])
        tensor_symbol = S1

    elif re.match(r"s2_\{[a-z][a-z]\}", symbol):
        # s2_{IJ}
        indices = tuple(symbol[4:6])
        tensor_symbol = S2

    elif re.match(r"lu11_\{[a-z][a-z][a-z]\}", symbol):
        # lu11_{Iij}
        indices = tuple(symbol[5:8])
        tensor_symbol = LU11

    elif re.match(r"lu12_\{[a-z][a-z][a-z][a-z]\}", symbol):
        # lu12_{IJij}
        indices = tuple(symbol[5:9])
        tensor_symbol = LU12

    elif re.match(r"u11_\{[a-z][a-z][a-z]\}", symbol):
        # u11_{Iij}
        indices = tuple(symbol[5:8])
        tensor_symbol = U11

    elif re.match(r"u12_\{[a-z][a-z][a-z][a-z]\}", symbol):
        # u12_{IJij}
        indices = tuple(symbol[5:9])
        tensor_symbol = U12

    elif re.match(r"delta_\{[a-z][a-z]\}", symbol):
        # delta_{ij}
        indices = tuple(symbol[7:9])
        tensor_symbol = Delta

    else:
        raise ValueError(f"Unknown symbol {symbol}")

    # Convert the indices to SpinIndex
    indices = tuple(
        SpinIndex(index, index_spins[index]) if index in index_spins else index for index in indices
    )

    return tensor_symbol[indices]


def prod(iterable):
    """Return the product of an iterable."""
    p = 1
    for i in iterable:
        p *= i
    return p


def get_factor(*indices):
    """Return the factor for a given operator, where a factor
    1/2 is raised to the power of the number of identical
    indices.
    """

    def factorial(n):
        if n in (0, 1):
            return 1
        elif n > 1:
            return n * factorial(n - 1)
        else:
            raise ValueError("{n}!".format(n=n))

    counts = {"occ": 0, "vir": 0, "nm": 0}

    for index in indices:
        counts[index.space] += 1

    n = 1
    for count in counts.values():
        n *= factorial(count)

    return Fraction(1, n)


def get_rank(rank=("SD", "", "")):
    """Return ranks for each string."""

    values = {
        "S": 1,
        "D": 2,
        "T": 3,
        "Q": 4,
    }

    return tuple(tuple(values[i] for i in j) for j in rank)


def get_hamiltonian(rank=("SD", "", ""), compress=False):
    """Define the core Hamiltonian."""

    # fermions
    h1e = one_e("f", ["occ", "vir"], norder=True)
    h2e = two_e("v", ["occ", "vir"], norder=True, compress=compress)
    h = h1e + h2e

    # bosons
    if rank[1] or rank[2]:
        hp = two_p("w") + one_p("G")
        hep = ep11("g", ["occ", "vir"], ["nm"], norder=True, name2="gc")
        h += hp + hep

    return h


def get_bra_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define left projection spaces."""

    rank = get_rank(rank)
    bras = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        operators = [FOperator(i, True) for i in occ] + [FOperator(a, False) for a in vir[::-1]]
        tensors = [Tensor(occ + vir, "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        operators = [BOperator(x, False) for x in nm]
        tensors = [Tensor(nm, "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    # fermion-boson coupling
    for n in rank[2]:
        i = Idx(0, "occ") if occs is None else occs[0]
        a = Idx(0, "vir") if virs is None else virs[0]
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        operators = [BOperator(x, False) for x in nm] + [FOperator(i, True), FOperator(a, False)]
        tensors = [Tensor(nm + [i, a], "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    return tuple(bras)


def get_bra_ip_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define left IP projection spaces."""

    rank = get_rank(rank)
    bras = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n - 1)] if virs is None else virs[: n - 1]
        operators = [FOperator(i, True) for i in occ] + [FOperator(a, False) for a in vir[::-1]]
        tensors = [Tensor(occ + vir, "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(bras)


def get_bra_ea_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define left IP projection spaces."""

    rank = get_rank(rank)
    bras = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n - 1)] if occs is None else occs[: n - 1]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        operators = [FOperator(i, True) for i in occ] + [FOperator(a, False) for a in vir[::-1]]
        tensors = [Tensor(vir + occ, "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(bras)


def get_ket_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define right projection spaces."""

    rank = get_rank(rank)
    kets = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        tensors = [Tensor(occ + vir, "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        operators = [BOperator(x, True) for x in nm]
        tensors = [Tensor(nm, "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    # fermion-boson coupling
    for n in rank[2]:
        i = Idx(0, "occ") if occs is None else occs[0]
        a = Idx(0, "vir") if virs is None else virs[0]
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        operators = [BOperator(x, True) for x in nm] + [FOperator(a, True), FOperator(i, False)]
        tensors = [Tensor(nm + [i, a], "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    return tuple(kets)


def get_ket_ip_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define left IP projection spaces."""

    rank = get_rank(rank)
    kets = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n - 1)] if virs is None else virs[: n - 1]
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        tensors = [Tensor(occ + vir, "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(kets)


def get_ket_ea_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define left IP projection spaces."""

    rank = get_rank(rank)
    kets = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n - 1)] if occs is None else occs[: n - 1]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ]
        tensors = [Tensor(vir + occ, "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(kets)


def get_r_ip_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define space of trial vector to apply an IP hamiltonian to."""

    rank = get_rank(rank)
    rs = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n - 1)] if virs is None else virs[: n - 1]
        name = "r{n}".format(n=n)
        scalar = get_factor(*occ, *vir)
        sums = [Sigma(i) for i in occ] + [Sigma(a) for a in vir]
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        tensors = [Tensor(occ + vir, name)]
        rs.append(Expression([Term(scalar, sums, tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(rs)


def get_r_ea_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define space of trial vector to apply an EA hamiltonian to."""

    rank = get_rank(rank)
    rs = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n - 1)] if occs is None else occs[: n - 1]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        name = "r{n}".format(n=n)
        scalar = get_factor(*occ, *vir)
        sums = [Sigma(a) for a in vir] + [Sigma(i) for i in occ]
        operators = [FOperator(a, True) for a in vir[::-1]] + [FOperator(i, False) for i in occ]
        tensors = [Tensor(vir + occ, name)]
        rs.append(Expression([Term(scalar, sums, tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(rs)


def get_r_ee_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define space of trial vector to apply an EE hamiltonian to."""

    rank = get_rank(rank)
    rs = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        name = "ree{n}".format(n=n)
        scalar = get_factor(*occ, *vir)
        sums = [Sigma(i) for i in occ] + [Sigma(a) for a in vir]
        operators = [FOperator(a, True) for a in vir[::-1]] + [FOperator(i, False) for i in occ]
        tensors = [Tensor(occ + vir, name)]
        rs.append(Expression([Term(scalar, sums, tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(rs)


def get_symm(dims, antis):
    """Get symmetry information in the `wick` format."""
    all_perms = []
    all_signs = []
    for tup in itertools.product(*[antisymmetric_permutations(dim) for dim in dims]):
        perms = [x.permutation for x in tup]
        signs = [x.sign for x in tup]
        signs = [sign if anti else 1 for sign, anti in zip(signs, antis)]
        perms = list(perms)
        n = 0
        for i in range(len(perms)):
            perms[i] = tuple(n + j for j in perms[i])
            n += len(perms[i])
        perm = sum(perms, tuple())
        sign = prod(signs)
        all_perms.append(list(perm))
        all_signs.append(sign)
    return TensorSym(all_perms, all_signs)


def get_excitation_ansatz(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define excitation amplitudes for the given ansatz."""

    rank = get_rank(rank)
    t = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        scalar = get_factor(*occ, *vir)
        sums = [Sigma(i) for i in occ] + [Sigma(a) for a in vir]
        name = "t{n}".format(n=n)
        tensors = [Tensor(occ + vir, name, sym=get_symm([n, n], [True, True]))]
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        t.append(Term(scalar, sums, tensors, operators, []))

    # boson
    for n in rank[1]:
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        scalar = get_factor(*nm)
        sums = [Sigma(x) for x in nm]
        name = "s{n}".format(n=n)
        tensors = [Tensor(nm, name, sym=get_symm([n], [False]))]
        operators = [BOperator(x, True) for x in nm]
        t.append(Term(scalar, sums, tensors, operators, []))

    # fermion-boson coupling
    for n in rank[2]:
        i = Idx(0, "occ") if occs is None else occs[0]
        a = Idx(0, "vir") if virs is None else virs[0]
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        scalar = get_factor(i, a, *nm)
        sums = [Sigma(x) for x in nm] + [Sigma(i), Sigma(a)]
        name = "u1{n}".format(n=n)
        tensors = [Tensor(nm + [i, a], name, sym=get_symm([n, 1, 1], [False, True, True]))]
        operators = [BOperator(x, True) for x in nm] + [FOperator(a, True), FOperator(i, False)]
        t.append(Term(scalar, sums, tensors, operators, []))

    return Expression(t)


def get_deexcitation_ansatz(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define de-excitation amplitudes for the given ansatz."""

    rank = get_rank(rank)
    l = []

    # fermion
    for n in rank[0]:
        # Swapped variables names so I can copy the code:
        vir = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        occ = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        scalar = get_factor(*occ, *vir)
        sums = [Sigma(i) for i in occ] + [Sigma(a) for a in vir]
        name = "l{n}".format(n=n)
        tensors = [Tensor(occ + vir, name, sym=get_symm([n, n], [True, True]))]
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        l.append(Term(scalar, sums, tensors, operators, []))

    # boson
    for n in rank[1]:
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        scalar = get_factor(*nm)
        sums = [Sigma(x) for x in nm]
        name = "ls{n}".format(n=n)
        tensors = [Tensor(nm, name, sym=get_symm([n], [False]))]
        operators = [BOperator(x, False) for x in nm]
        l.append(Term(scalar, sums, tensors, operators, []))

    # fermion-boson coupling
    for n in rank[2]:
        # Swapped variables names so I can copy the code:
        a = Idx(0, "occ") if occs is None else occs[0]
        i = Idx(0, "vir") if virs is None else virs[0]
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        scalar = get_factor(i, a, *nm)
        sums = [Sigma(x) for x in nm] + [Sigma(i), Sigma(a)]
        name = "lu1{n}".format(n=n)
        tensors = [Tensor(nm + [i, a], name, sym=get_symm([n, 1, 1], [False, True, True]))]
        operators = [BOperator(x, False) for x in nm] + [FOperator(a, True), FOperator(i, False)]
        l.append(Term(scalar, sums, tensors, operators, []))

    return Expression(l)


def bch(h, t, max_commutator=4):
    r"""Construct successive orders of \bar{H} and return a list."""

    def factorial(n):
        if n in (0, 1):
            return 1
        elif n > 1:
            return n * factorial(n - 1)
        else:
            raise ValueError("{n}!".format(n=n))

    comms = [h]
    for i in range(max_commutator):
        comms.append(commute(comms[-1], t))

    hbars = [h]
    for i in range(1, len(comms)):
        scalar = Fraction(1, factorial(i))
        hbars.append(hbars[-1] + comms[i] * scalar)

    return hbars


construct_hbar = bch
