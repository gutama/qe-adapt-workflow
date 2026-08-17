"""Electronic state and band structure canonical models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qeanalyzer.io.qe_xml import HARTREE_TO_EV, HARTREE_TO_RY


@dataclass
class QEElectronicState:
    """Complete electronic structure state from SCF/NSCF/XML/out results."""

    total_energy_ry: float | None = None
    total_energy_ev: float | None = None
    total_energy_hartree: float | None = None

    one_electron_ry: float | None = None
    hartree_energy_ry: float | None = None
    xc_energy_ry: float | None = None
    ewald_energy_ry: float | None = None
    smearing_energy_ry: float | None = None

    fermi_energy_ev: float | None = None
    highest_occupied_ev: float | None = None
    lowest_unoccupied_ev: float | None = None
    band_gap_ev: float | None = None

    n_electrons: float | None = None
    n_bands: int | None = None
    n_kpoints: int | None = None
    lsda: bool = False
    noncolin: bool = False
    spinorbit: bool = False

    total_magnetization: float | None = None
    absolute_magnetization: float | None = None
    is_metal: bool | None = None

    eigenvalues_ev: list[list[float]] = field(default_factory=list)
    occupations: list[list[float]] = field(default_factory=list)

    @classmethod
    def from_energies(
        cls,
        total_energy_ry: float | None = None,
        total_energy_hartree: float | None = None,
        one_electron_ry: float | None = None,
        hartree_energy_ry: float | None = None,
        xc_energy_ry: float | None = None,
        ewald_energy_ry: float | None = None,
        smearing_energy_ry: float | None = None,
        fermi_energy_ev: float | None = None,
        highest_occupied_ev: float | None = None,
        lowest_unoccupied_ev: float | None = None,
        n_electrons: float | None = None,
        n_bands: int | None = None,
        n_kpoints: int | None = None,
        total_magnetization: float | None = None,
        absolute_magnetization: float | None = None,
        eigenvalues_ev: list[list[float]] | None = None,
        occupations: list[list[float]] | None = None,
        lsda: bool = False,
        noncolin: bool = False,
        spinorbit: bool = False,
    ) -> QEElectronicState:
        """Construct electronic state with automatic unit derivations."""
        tot_ry = total_energy_ry
        tot_ha = total_energy_hartree
        tot_ev = None

        if tot_ry is not None:
            tot_ha = tot_ry / HARTREE_TO_RY
            tot_ev = tot_ha * HARTREE_TO_EV
        elif tot_ha is not None:
            tot_ry = tot_ha * HARTREE_TO_RY
            tot_ev = tot_ha * HARTREE_TO_EV

        gap_ev = None
        if highest_occupied_ev is not None and lowest_unoccupied_ev is not None:
            gap_ev = max(0.0, lowest_unoccupied_ev - highest_occupied_ev)

        metal = None
        if fermi_energy_ev is not None and highest_occupied_ev is None:
            metal = True
        elif gap_ev is not None:
            metal = gap_ev < 1e-4

        return cls(
            total_energy_ry=tot_ry,
            total_energy_ev=tot_ev,
            total_energy_hartree=tot_ha,
            one_electron_ry=one_electron_ry,
            hartree_energy_ry=hartree_energy_ry,
            xc_energy_ry=xc_energy_ry,
            ewald_energy_ry=ewald_energy_ry,
            smearing_energy_ry=smearing_energy_ry,
            fermi_energy_ev=fermi_energy_ev,
            highest_occupied_ev=highest_occupied_ev,
            lowest_unoccupied_ev=lowest_unoccupied_ev,
            band_gap_ev=gap_ev,
            n_electrons=n_electrons,
            n_bands=n_bands,
            n_kpoints=n_kpoints,
            lsda=lsda,
            noncolin=noncolin,
            spinorbit=spinorbit,
            total_magnetization=total_magnetization,
            absolute_magnetization=absolute_magnetization,
            is_metal=metal,
            eigenvalues_ev=eigenvalues_ev or [],
            occupations=occupations or [],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert electronic state to JSON-serializable dictionary."""
        return {
            "total_energy_ry": self.total_energy_ry,
            "total_energy_ev": self.total_energy_ev,
            "total_energy_hartree": self.total_energy_hartree,
            "energy_components_ry": {
                "one_electron": self.one_electron_ry,
                "hartree": self.hartree_energy_ry,
                "xc": self.xc_energy_ry,
                "ewald": self.ewald_energy_ry,
                "smearing": self.smearing_energy_ry,
            },
            "fermi_energy_ev": self.fermi_energy_ev,
            "highest_occupied_ev": self.highest_occupied_ev,
            "lowest_unoccupied_ev": self.lowest_unoccupied_ev,
            "band_gap_ev": self.band_gap_ev,
            "n_electrons": self.n_electrons,
            "n_bands": self.n_bands,
            "n_kpoints": self.n_kpoints,
            "is_metal": self.is_metal,
            "spin": {
                "lsda": self.lsda,
                "noncolin": self.noncolin,
                "spinorbit": self.spinorbit,
                "total_magnetization": self.total_magnetization,
                "absolute_magnetization": self.absolute_magnetization,
            },
            "eigenvalues_ev": self.eigenvalues_ev,
            "occupations": self.occupations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QEElectronicState:
        """Construct QEElectronicState from dictionary."""
        e_comps = data.get("energy_components_ry", {})
        spin = data.get("spin", {})
        return cls(
            total_energy_ry=data.get("total_energy_ry"),
            total_energy_ev=data.get("total_energy_ev"),
            total_energy_hartree=data.get("total_energy_hartree"),
            one_electron_ry=e_comps.get("one_electron"),
            hartree_energy_ry=e_comps.get("hartree"),
            xc_energy_ry=e_comps.get("xc"),
            ewald_energy_ry=e_comps.get("ewald"),
            smearing_energy_ry=e_comps.get("smearing"),
            fermi_energy_ev=data.get("fermi_energy_ev"),
            highest_occupied_ev=data.get("highest_occupied_ev"),
            lowest_unoccupied_ev=data.get("lowest_unoccupied_ev"),
            band_gap_ev=data.get("band_gap_ev"),
            n_electrons=data.get("n_electrons"),
            n_bands=data.get("n_bands"),
            n_kpoints=data.get("n_kpoints"),
            is_metal=data.get("is_metal"),
            lsda=spin.get("lsda", False),
            noncolin=spin.get("noncolin", False),
            spinorbit=spin.get("spinorbit", False),
            total_magnetization=spin.get("total_magnetization"),
            absolute_magnetization=spin.get("absolute_magnetization"),
            eigenvalues_ev=data.get("eigenvalues_ev", []),
            occupations=data.get("occupations", []),
        )
