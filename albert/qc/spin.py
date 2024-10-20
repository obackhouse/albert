"""Spin integration routines.
"""

import itertools
from collections import defaultdict
from numbers import Number

from albert.algebra import Mul, Add
from albert.qc.ghf import GHFTensor
from albert.qc.rhf import RHFTensor
from albert.qc.uhf import UHFTensor


def generalised_to_unrestricted(expr, target_restricted=False):
    """
    Convert an expression in a generalised basis to an unrestricted
    basis.

    Parameters
    ----------
    expr : Base
        The expression to convert.
    target_restricted : bool, optional
        Whether the function is being called with the intent to convert
        to restricted, which sometimes requires different behaviour.
        Default value is `False`.

    Returns
    -------
    expr_uhf : Base
        The expression in the unrestricted basis.
    """

    # Loop over addition layer
    new_exprs = defaultdict(int)
    for mul_args in expr.nested_view():
        # Loop over multiplication layer
        non_spin_args = []
        spin_args = []
        for arg in mul_args:
            # If the argument is a number or is already unrestricted,
            # it doesn't change
            if isinstance(arg, (Number, UHFTensor)):
                non_spin_args.append(arg)
                continue

            # At this point, the argument must be a GHFTensor
            if not isinstance(arg, GHFTensor):
                raise ValueError(
                    "`generalised_to_unrestricted` requires expressions"
                    " consisting of types `GHFTensor`, `UHFTensor`, and"
                    " `Number`. Got type `{}`.".format(type(arg))
                )

            # Convert to the spin cases
            spin_args.append(arg.as_uhf(target_restricted=target_restricted))

        # Separate the spin cases
        for spin_args_perm in itertools.product(*spin_args):
            # Check that the indices for this permutation have
            # consistent spins
            index_spins = {}
            for index in itertools.chain(*(arg.external_indices for arg in spin_args_perm)):
                if index.spin:
                    if index.name not in index_spins:
                        index_spins[index.name] = index.spin
                    elif index_spins[index.name] != index.spin:
                        break
            else:
                # This contribution is good, add it to the new
                # expression for the relevant indices
                new_mul = Mul(*non_spin_args, *spin_args_perm)
                if isinstance(new_exprs[new_mul.external_indices], Add):
                    # avoid too much recursion
                    new_exprs[new_mul.external_indices] = Add(*new_exprs[new_mul.external_indices].args, new_mul)
                else:
                    new_exprs[new_mul.external_indices] += new_mul

    ## Partially canonicalise
    #canon_exprs = defaultdict(int)
    #for part in tuple(new_exprs.values()):
    #    # FIXME we should try to avoid this canonicalisation
    #    summed_part = 0
    #    for arg in part.nested_view():
    #        if isinstance(arg, Add):
    #            summed_part = Add(*summed_part.args, Mul(*arg))
    #        else:
    #            summed_part += Mul(*arg)
    #    canon_exprs[part.external_indices] += part
    #new_exprs = canon_exprs

    return tuple(new_exprs.values())


def _unrestricted_to_restricted(expr):
    """
    Convert an expression in a unrestricted basis to a restricted basis.

    Parameters
    ----------
    expr : Base
        The expression to convert.

    Returns
    -------
    expr_rhf : Base
        The expression in the restricted basis.
    """

    # Loop over addition layer
    new_expr = 0
    for mul_args in expr.nested_view():
        # Loop over multiplication layer
        args = []
        for arg in mul_args:
            # If the argument is a number or is already restricted,
            # it doesn't change
            if isinstance(arg, (Number, RHFTensor)):
                args.append(arg)
                continue

            # At this point, the argument must be a UHFTensor
            if not isinstance(arg, UHFTensor):
                raise ValueError(
                    "`unrestricted_to_restricted` requires expressions"
                    " consisting of types `UHFTensor`, `RHFTensor`, and"
                    " `Number`. Got type `{}`.".format(type(arg))
                )

            # Convert to the spin cases
            args.append(arg.as_rhf())

        # Add the new term to the expression
        new_expr += Mul(*args)

    return new_expr


def generalised_to_restricted(expr):
    """
    Convert an expression in a generalised basis to a restricted
    basis.

    Parameters
    ----------
    expr : Base
        The expression to convert.

    Returns
    -------
    expr_rhf : Base
        The expression in the restricted basis.
    """

    # Convert to unrestricted
    expr_uhf = generalised_to_unrestricted(expr, target_restricted=True)

    # Partially canonicalise
    parts = defaultdict(int)
    for part in expr_uhf:
        part = sum(Mul(*arg).canonicalise() for arg in part.nested_view())
        parts[part.external_indices] += part
    expr_uhf = tuple(parts.values())

    # Convert to restricted
    expr_rhf = sum(_unrestricted_to_restricted(e) for e in expr_uhf)

    return expr_rhf
