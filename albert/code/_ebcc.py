"""Class for code generation using `einsum` for `ebcc`."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from albert.code.einsum import EinsumCodeGenerator
from albert.misc import ExclusionSet

if TYPE_CHECKING:
    from typing import Optional

    from albert.tensor import Tensor


_amplitude_names = [
    "t1",
    "t2",
    "t3",
    "t4",
    "t4a",
    "l1",
    "l2",
    "l3",
    "l4",
    "l4a",
    "s1",
    "s2",
    "ls1",
    "ls2",
    "u11",
    "u12",
    "lu11",
    "lu12",
    "r1",
    "r2",
    "r3",
]
_descriptions = {
    "f": "Fock matrix.",
    "v": "Electron repulsion integrals.",
    "G": "One-boson Hamiltonian.",
    "w": "Two-boson Hamiltonian.",
    "g": "Electron-boson coupling.",
    "e_cc": "Coupled cluster energy.",
    "e_pert": "Perturbation energy.",
    "rdm1": "One-particle reduced density matrix.",
    "rdm2": "Two-particle reduced density matrix.",
    "rdm1_b": "One-body reduced density matrix.",
    "rdm_eb_cre": "Electron-boson coupling reduced density matrix, creation part.",
    "rdm_eb_des": "Electron-boson coupling reduced density matrix, annihilation part.",
    "dm_cre": "Single boson density matrix, creation part.",
    "dm_des": "Single boson density matrix, annihilation part.",
    **{f"{name}": f"{name.upper()} amplitudes." for name in _amplitude_names},
    **{f"{name}new": f"Updated {name.upper()} residuals." for name in _amplitude_names},
}


class EBCCCodeGenerator(EinsumCodeGenerator):
    """Class for code generation using `einsum` for `ebcc`."""

    _einsum_kwargs = {}
    _add_spaces = ExclusionSet(_amplitude_names + [f"{name}new" for name in _amplitude_names])

    _einsum = "einsum"
    _namespace = "Namespace"

    def get_name(
        self,
        tensor: Tensor,
        add_spins: Optional[bool] = None,
        add_spaces: Optional[bool] = None,
        spin_delimiter: str = ".",
        space_delimiter: str = ".",
    ) -> str:
        """Get the name for a tensor in the code.

        Args:
            tensor: The tensor.
            add_spins: Whether to add spins to the name.
            add_spaces: Whether to add spaces to the name.
            spin_delimiter: The delimiter for spins.
            space_delimiter: The delimiter for spaces.
        """
        if tensor.name is None:
            raise ValueError("Can't generate a name for a tensor without the `name` attribute.")
        string: str = tensor.name

        # Add the spins
        _add_spins = (add_spins is None and tensor.name in self._add_spins) or add_spins
        _add_spins = _add_spins and not tensor.name.startswith("tmp")
        _add_spins = _add_spins and not tensor.name.startswith("ints")
        _add_spins = _add_spins and any(i.spin in ("a", "b") for i in tensor.external_indices)
        if _add_spins:
            spins = tuple(i.spin for i in tensor.external_indices if i.spin in ("a", "b"))
            if len(spins):
                string += spin_delimiter + "".join(spins)

        # Add the spaces
        _add_spaces = (add_spaces is None and tensor.name in self._add_spaces) or add_spaces
        _add_spaces = _add_spaces and not tensor.name.startswith("tmp")
        _add_spaces = _add_spaces and not tensor.name.startswith("ints")
        _add_spaces = _add_spaces and all(i.space for i in tensor.external_indices)
        if _add_spaces:
            spaces = tuple(cast(str, i.space) for i in tensor.external_indices)
            if len(spaces):
                string += space_delimiter + "".join(spaces)

        return string

    def get_argument(self, arg: Tensor) -> str:
        """Get the argument string.

        Args:
            arg: The tensor.

        Returns:
            The argument string.
        """
        name = self.get_name(arg, add_spins=False, add_spaces=False)
        if name.startswith("ints"):
            return "ints"
        return name

    def module_imports(self) -> None:
        """Write the module imports."""
        self.write("from ebcc import numpy as np")
        self.write("from ebcc.util import pack_2e, einsum, dirsum, Namespace")

    def tensor_cleanup(self, *args: Tensor) -> None:
        """Write tensor cleanup.

        Args:
            args: The tensors.
        """
        names = sorted(
            set(
                self.get_name(arg, add_spins=False, add_spaces=False)
                for arg in args
                if not (
                    any(arg.name == name for name, _, _ in self._tensor_cleaned)
                    or arg.name.startswith("ints")
                )
            )
        )
        if names:
            self.write("del " + ", ".join(names))
