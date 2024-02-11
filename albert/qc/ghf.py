"""Expressions for generalised bases.
"""

from albert.qc.rhf import _make_symmetry
from albert.symmetry import Permutation, Symmetry
from albert.tensor import Symbol


def antisymmetric_permutations(n):
    """
    Return permutations of `n` objects with a sign equal to +1 for an
    even number of swaps and -1 for an odd number of swaps.

    Parameters
    ----------
    n : int
        Number of objects.

    Returns
    -------
    perms : list of Permutation
        Permutations.
    """

    def _permutations(seq):
        if not seq:
            return [[]]

        items = []
        for i, item in enumerate(_permutations(seq[:-1])):
            inds = range(len(item) + 1)
            if i % 2 == 0:
                inds = reversed(inds)
            items += [item[:i] + seq[-1:] + item[i:] for i in inds]

        return items

    return [
        Permutation(item, -1 if i % 2 else 1)
        for i, item in enumerate(_permutations(list(range(n))))
    ]


class Hamiltonian1e(Symbol):
    """Constructor for one-electron Hamiltonian-like symbols."""

    DESIRED_RANK = 2

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        self.symmetry = _make_symmetry(
            (0, 1),
            (1, 0),
        )

    def get_as_uhf(self, *indices):
        """Get the corresponding tensor for an unrestricted basis."""
        indices_α = tuple((idx, "α") for idx in indices)
        indices_β = tuple((idx, "β") for idx in indices)
        uhf_symbol = getattr(uhf, self.__class__.__name__)
        return uhf_symbol[indices_α] + uhf_symbol[indices_β]


Fock = Hamiltonian1e("f")


class Hamiltonian2e(Symbol):
    """Constructor for two-electron Hamiltonian-like symbols."""

    DESIRED_RANK = 4

    def __init__(self, name):
        """Initialise the object."""
        self.name = name
        # FIXME this is for real orbitals only
        self.symmetry = Symmetry(
            (Permutation((0, 1, 2, 3), +1)),
            (Permutation((0, 1, 3, 2), -1)),
            (Permutation((1, 0, 2, 3), -1)),
            (Permutation((1, 0, 3, 2), +1)),
            (Permutation((2, 3, 0, 1), +1)),
            (Permutation((3, 2, 0, 1), -1)),
            (Permutation((2, 3, 1, 0), -1)),
            (Permutation((3, 2, 1, 0), +1)),
        )


ERI = Hamiltonian2e("v")


class FermionicAmplitude(Symbol):
    """Constructor for amplitude symbols."""

    def __init__(self, name, num_covariant, num_contravariant):
        """Initialise the object."""
        self.name = name
        self.DEISRED_RANK = num_covariant + num_contravariant
        perms = []
        for perm_covariant in antisymmetric_permutations(num_covariant):
            for perm_contravariant in antisymmetric_permutations(num_contravariant):
                perms.append(perm_covariant + perm_contravariant)
        self.symmetry = Symmetry(*perms)


T1 = FermionicAmplitude("t1", 1, 1)
T2 = FermionicAmplitude("t2", 2, 2)
T3 = FermionicAmplitude("t3", 3, 3)
