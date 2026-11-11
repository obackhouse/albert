"""Options for chemistry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from albert._core import Permutation
from albert.options import PhaseFactor


class ReferenceType(StrEnum):
    """Reference type."""

    RHF = "rhf"
    UHF = "uhf"
    GHF = "ghf"

    @property
    def prefix(self) -> str:
        """The prefix for this reference type."""
        return self.value[0]


class OrbitalType(StrEnum):
    """Orbital type."""

    ALPHA = "alpha"
    BETA = "beta"
    SPIN = "spin"
    SPATIAL = "spatial"

    @property
    def subscript(self) -> str:
        r"""The subscript part of an index symbol for this orbital type.

        Note:
            The character convention uses Greek subscripts for orbitals of a specific spin, and a
            :math:`\gamma` for spin orbitals. Spatial orbitals do not get a subscript.
        """
        if self is OrbitalType.ALPHA:
            return "_α"
        elif self is OrbitalType.BETA:
            return "_β"
        elif self is OrbitalType.SPIN:
            return "_γ"
        elif self is OrbitalType.SPATIAL:
            return ""
        raise ValueError(f"unknown orbital type: {self}")

    @property
    def size_factor(self) -> int:
        """The size factor for this orbital type."""
        if self is OrbitalType.SPIN:
            return 2
        return 1


@dataclass(frozen=True)
class TensorSymbol:
    """Tensor symbol for chemistry."""

    symbol: str
    """Symbol for the tensor."""

    description: str
    """Description of the tensor."""

    rank: int
    """Rank of the tensor."""

    symmetries: list[Permutation] | None = field(default=None)
    """Symmetries of the tensor for different reference types."""


def _s(*keys: str) -> list[Permutation]:
    """Return the list of permutations corresponding to the given keys."""
    permutations = []
    base = keys[0].lstrip("+-")
    for key in keys:
        sign = 1
        chars = key
        if key.startswith("-"):
            sign = -1
            chars = chars[1:]
        elif key.startswith("+"):
            chars = chars[1:]
        perm = tuple(base.index(c) for c in chars)
        permutations.append(Permutation(perm, PhaseFactor(sign)))
    return permutations


HERMITIAN_SYMMETRY = _s("+ij", "+ji")
ERI_SYMMETRY = _s("+ijkl", "jilk", "klij", "lkji")
T2_SYMMETRY = _s("+ijab", "+jiba")
T3_SYMMETRY = _s("+ijkabc", "+kjicba")

ERI_ANTISYMMETRY = _s("+ijkl", "-jikl", "-ijlk", "+jilk", "-klij", "+lkij", "+klji", "-lkji")
T2_ANTISYMMETRY = _s("+ijab", "-jiab", "-ijba", "+jiba")
T3_ANTISYMMETRY = _s("+ijkabc", "-jikabc", "-ikjcba", "+kjicba")
D2_ANTISYMMETRY = _s("+ijkl", "-jikl", "-ijlk", "+jilk")

SUPPORTED_TENSOR_SYMBOLS = {
    # Hamiltonian elements:
    "h": TensorSymbol("h", "One-electron Hamiltonian", 2, symmetries=HERMITIAN_SYMMETRY),
    "f": TensorSymbol("f", "Fock matrix", 2, symmetries=HERMITIAN_SYMMETRY),
    "v": TensorSymbol("v", "Two-electron integrals", 4, symmetries=ERI_SYMMETRY),
    "v_a": TensorSymbol("v_a", "Antisymmetrized two-electron integrals", 4, symmetries=ERI_ANTISYMMETRY),
    # Coupled cluster amplitudes:
    "t1": TensorSymbol("t1", "Single excitation amplitudes", 2),
    "t2": TensorSymbol("t2", "Double excitation amplitudes", 4, symmetries=T2_SYMMETRY),
    "t3": TensorSymbol("t3", "Triple excitation amplitudes", 6, symmetries=T3_SYMMETRY),
    "t2_a": TensorSymbol("t2_a", "Antisymmetrized double excitation amplitudes", 4, symmetries=T2_ANTISYMMETRY),
    "t3_a": TensorSymbol("t3_a", "Antisymmetrized triple excitation amplitudes", 6, symmetries=T3_ANTISYMMETRY),
    "l1": TensorSymbol("l1", "Single de-excitation amplitudes", 2),
    "l2": TensorSymbol("l2", "Double de-excitation amplitudes", 4, symmetries=T2_SYMMETRY),
    "l3": TensorSymbol("l3", "Triple de-excitation amplitudes", 6, symmetries=T3_SYMMETRY),
    "l2_a": TensorSymbol("l2_a", "Antisymmetrized double de-excitation amplitudes", 4, symmetries=T2_ANTISYMMETRY),
    "l3_a": TensorSymbol("l3_a", "Antisymmetrized triple de-excitation amplitudes", 6, symmetries=T3_ANTISYMMETRY),
    # Equation-of-motion amplitudes:
    "r0": TensorSymbol("r0", "Reference amplitude", 0),
    "r1ip": TensorSymbol("r1ip", "Single ionization amplitudes", 1),
    "r2ip": TensorSymbol("r2ip", "Double ionization amplitudes", 3),
    "r3ip": TensorSymbol("r3ip", "Triple ionization amplitudes", 5),
    "r1ea": TensorSymbol("r1ea", "Single electron attachment amplitudes", 1),
    "r2ea": TensorSymbol("r2ea", "Double electron attachment amplitudes", 3),
    "r3ea": TensorSymbol("r3ea", "Triple electron attachment amplitudes", 5),
    "r1ee": TensorSymbol("r1ee", "Single electron excitation amplitudes", 2),
    "r2ee": TensorSymbol("r2ee", "Double electron excitation amplitudes", 4),
    "r3ee": TensorSymbol("r3ee", "Triple electron excitation amplitudes", 6),
    # Equation-of-motion left-hand amplitudes:
    "l0": TensorSymbol("l0", "Reference left-hand amplitude", 0),
    "l1ip": TensorSymbol("l1ip", "Single left-hand ionization amplitudes", 1),
    "l2ip": TensorSymbol("l2ip", "Double left-hand ionization amplitudes", 3),
    "l3ip": TensorSymbol("l3ip", "Triple left-hand ionization amplitudes", 5),
    "l1ea": TensorSymbol("l1ea", "Single left-hand electron attachment amplitudes", 1),
    "l2ea": TensorSymbol("l2ea", "Double left-hand electron attachment amplitudes", 3),
    "l3ea": TensorSymbol("l3ea", "Triple left-hand electron attachment amplitudes", 5),
    "l1ee": TensorSymbol("l1ee", "Single left-hand electron excitation amplitudes", 2),
    "l2ee": TensorSymbol("l2ee", "Double left-hand electron excitation amplitudes", 4),
    "l3ee": TensorSymbol("l3ee", "Triple left-hand electron excitation amplitudes", 6),
    # Density matrices:
    "d1": TensorSymbol("d1", "One-particle density matrix", 2, symmetries=HERMITIAN_SYMMETRY),
    "d2": TensorSymbol("d2", "Two-particle density matrix", 4),
    "d2_a": TensorSymbol("d2_a", "Antisymmetrized two-particle density matrix", 4, symmetries=D2_ANTISYMMETRY),
}
