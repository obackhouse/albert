"""Spin integration routines."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, cast

from albert.algebra import _compose_mul
from albert.index import Index
from albert.qc.tensor import QTensor
from albert.scalar import Scalar

if TYPE_CHECKING:
    from typing import Optional

    from albert.base import Base


def ghf_to_uhf(
    expr: Base,
    target_rhf: bool = False,
    canonicalise: bool = True,
) -> tuple[Base, ...]:
    """Convert the spin orbital indices of an expression to indices with spin.

    Args:
        expr: The expression to convert.
        target_rhf: Whether the target is RHF. For some tensors, the intermediate conversion to UHF
            is different depending on the target.
        canonicalise: Whether to canonicalise the tensors after converting them to UHF basis.

    Returns:
        Tuple of expressions resulting from the conversion.
    """
    # Expand the expression into an Add[Mul[Tensor | Scalar]]
    expr = expr.expand()

    # Loop over leaves of the expression tree
    new_exprs: dict[tuple[Index, ...], Base] = defaultdict(Scalar)
    for mul in expr.children:
        # Split the leaves into those that are already unrestricted and those that are not
        scalars = []
        tensors = []
        for leaf in mul.children:
            if isinstance(leaf, Scalar):
                scalars.append(leaf)
            else:
                leaf_as_uhf = cast(QTensor, leaf).as_uhf(target_rhf=target_rhf)
                if canonicalise:
                    leaf_as_uhf = tuple(e.canonicalise() for e in leaf_as_uhf)
                tensors.append(leaf_as_uhf)

        # Construct the unrestricted leaves
        for tensors_perm in itertools.product(*tensors):
            # Check the indices for this permutation have consistent spins
            index_spins = {}
            for index in itertools.chain(*(leaf.external_indices for leaf in tensors_perm)):
                if index.spin:
                    if index.name not in index_spins:
                        index_spins[index.name] = index.spin
                    elif index_spins[index.name] != index.spin:
                        break
            else:
                # This contribution is valid, add it to the new expression
                new_mul = _compose_mul(*scalars, *tensors_perm)
                new_exprs[new_mul.external_indices] += new_mul

    # Expand and collect
    new_exprs = tuple(expr.expand().collect() for expr in new_exprs.values())

    return new_exprs


def uhf_to_rhf(expr: Base, canonicalise: bool = True) -> Base:
    """Convert the indices with spin of an expression to restricted indices.

    Args:
        expr: The expression to convert.
        canonicalise: Whether to canonicalise the tensors after converting them to RHF basis.

    Returns:
        Tuple of expressions resulting from the conversion.
    """
    # Expand the expression into an Add[Mul[Tensor | Scalar]]
    expr = expr.expand()

    # Loop over leaves of the expression tree
    new_expr: Base = Scalar(0)
    for mul in expr.children:
        leaves: list[Base] = []
        for leaf in mul.children:
            if isinstance(leaf, Scalar):
                leaves.append(leaf)
            else:
                leaf_as_rhf = cast(QTensor, leaf).as_rhf()
                if canonicalise:
                    leaf_as_rhf = leaf_as_rhf.canonicalise()
                leaves.append(leaf_as_rhf)

        # Construct the new expression
        new_mul = _compose_mul(*leaves)
        new_expr += new_mul

    # Expand and collect
    new_expr = new_expr.expand().collect()

    return new_expr


def ghf_to_rhf(expr: Base, canonicalise: bool = True) -> Base:
    """Convert the indices with spin of an expression to restricted indices.

    Args:
        expr: The expression to convert.
        canonicalise: Whether to canonicalise the tensors after converting them between bases.

    Returns:
        Tuple of expressions resulting from the conversion.
    """
    # Convert to UHF
    uhf_exprs = ghf_to_uhf(expr, target_rhf=True, canonicalise=canonicalise)

    # Convert to RHF
    rhf_exprs = sum((uhf_to_rhf(expr, canonicalise=canonicalise) for expr in uhf_exprs), Scalar(0))

    return rhf_exprs


def get_amplitude_spins(
    occ_indices: list[Index | str],
    vir_indices: list[Index | str],
    spin_type: str,
) -> list[dict[str, Optional[str]]]:
    """Get the spins cases required for amplitudes in the given spin type.

    Args:
        occ_indices: The occupied indices.
        vir_indices: The virtual indices.
        spin_type: The spin type to consider.

    Returns:
        List of dictionaries mapping index names to spins, with list elements enumerating the cases.
    """
    if len(occ_indices) == len(vir_indices):
        no = nv = len(occ_indices)
    else:
        no = len(occ_indices)
        nv = len(vir_indices)

    def _get_name(index: Index | str) -> str:
        """Get the name of the index."""
        if isinstance(index, str):
            return index
        return index.name

    cases: list[dict[str, Optional[str]]] = []

    if spin_type == "rhf":
        # RHF
        case: dict[str, Optional[str]] = {}
        for i, index in enumerate(occ_indices):
            case[_get_name(index)] = ["a", "b"][i % 2]
        for i, index in enumerate(vir_indices):
            case[_get_name(index)] = ["a", "b"][i % 2]
        cases.append(case)
        # FIXME how to generalise?
        if len(occ_indices) == len(vir_indices) == 4:
            case = dict(zip(map(_get_name, occ_indices + vir_indices), ["a", "b", "a", "a"] * 2))
            cases.append(case)

    elif spin_type == "uhf":
        # UHF
        if no == nv:
            it = list(itertools.combinations_with_replacement(["a", "b"], no))
        else:
            it = list(itertools.product(["a", "b"], repeat=max(no, nv)))
        for spins in it:
            if no == nv:
                # Canonicalise the spin order
                best: tuple[tuple[str, ...], int] = (("",), int(1e10))
                for s in itertools.permutations(spins):
                    penalty = 0
                    if abs(spins.count("a") - spins.count("b")) < 2:
                        # Use alternating spins (ababab... or bababa...)
                        for i in range(len(s) - 1):
                            penalty += int(s[i] == s[i + 1]) * 2
                        if s[0] != min(s):
                            penalty += 1
                    else:
                        # Use lexicographic spins (abbbb... or aa...aab)
                        for i in range(len(s) - 1):
                            penalty += int(s[i] > s[i + 1])
                    if penalty < best[1]:
                        best = (s, penalty)
                spins = best[0]
            case = {}
            for i, spin in enumerate(spins):
                if i < no:
                    case[_get_name(occ_indices[i])] = spin
                if i < nv:
                    case[_get_name(vir_indices[i])] = spin
            cases.append(case)

    elif spin_type == "ghf":
        # GHF
        case = {}
        for i in range(no):
            case[_get_name(occ_indices[i])] = None
        for i in range(nv):
            case[_get_name(vir_indices[i])] = None
        cases.append(case)

    return cases


def get_density_spins(indices: list[Index | str], spin_type: str) -> list[dict[str, Optional[str]]]:
    """Get the spins cases required for density matrices in the given spin type.

    Args:
        indices: The indices.
        spin_type: The spin type to consider.

    Returns:
        List of dictionaries mapping index names to spins, with list elements enumerating the cases.
    """
    if len(indices) not in {2, 4}:
        raise ValueError("Density matrices must have 2 or 4 indices.")
    order = len(indices) // 2

    def _get_name(index: Index | str) -> str:
        """Get the name of the index."""
        if isinstance(index, str):
            return index
        return index.name

    cases: list[tuple[Optional[str], ...]]

    if spin_type == "rhf":
        # RHF
        if order == 1:
            cases = [("a", "a")]
        else:
            cases = [
                ("a", "a", "a", "a"),
                ("a", "b", "a", "b"),
                ("b", "a", "b", "a"),
                ("b", "b", "b", "b"),
            ]

    elif spin_type == "uhf":
        # UHF
        if order == 1:
            cases = [("a", "a"), ("b", "b")]
        else:
            cases = [
                ("a", "a", "a", "a"),
                ("a", "b", "a", "b"),
                ("b", "b", "b", "b"),
            ]

    elif spin_type == "ghf":
        # GHF
        if order == 1:
            cases = [(None, None)]
        else:
            cases = [(None, None, None, None)]

    return [dict(zip(map(_get_name, indices), case)) for case in cases]
