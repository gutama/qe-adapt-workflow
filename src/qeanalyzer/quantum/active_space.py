"""Active space selection for DFT to quantum solver (ADAPT-VQE) handoff."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qeanalyzer.models import QEElectronicState, QERunResult


@dataclass
class ActiveSpace:
    """Canonical record of a selected active space for post-DFT quantum solvers."""

    method: str
    active_orbitals: list[int] = field(default_factory=list)  # 0-indexed band indices
    n_active_orbitals: int = 0
    n_spin_orbitals: int = 0
    n_active_electrons: float = 0.0
    frozen_core_orbitals: list[int] = field(default_factory=list)
    frozen_virtual_orbitals: list[int] = field(default_factory=list)
    n_core_electrons: float = 0.0
    energy_window_ev: tuple[float, float] | None = None
    fermi_energy_ev: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and canonicalize active space parameters."""
        # Ensure active_orbitals are strictly sorted and unique
        self.active_orbitals = sorted(set(self.active_orbitals))
        self.frozen_core_orbitals = sorted(set(self.frozen_core_orbitals))
        self.frozen_virtual_orbitals = sorted(set(self.frozen_virtual_orbitals))

        if not self.n_active_orbitals:
            self.n_active_orbitals = len(self.active_orbitals)
        if not self.n_spin_orbitals:
            self.n_spin_orbitals = self.n_active_orbitals * 2

    def to_dict(self) -> dict[str, Any]:
        """Convert ActiveSpace to dictionary."""
        return {
            "method": self.method,
            "active_orbitals": list(self.active_orbitals),
            "n_active_orbitals": self.n_active_orbitals,
            "n_spin_orbitals": self.n_spin_orbitals,
            "n_active_electrons": self.n_active_electrons,
            "frozen_core_orbitals": list(self.frozen_core_orbitals),
            "frozen_virtual_orbitals": list(self.frozen_virtual_orbitals),
            "n_core_electrons": self.n_core_electrons,
            "energy_window_ev": list(self.energy_window_ev) if self.energy_window_ev else None,
            "fermi_energy_ev": self.fermi_energy_ev,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActiveSpace:
        """Construct ActiveSpace from dictionary."""
        win = data.get("energy_window_ev")
        energy_window_ev = tuple(win) if win is not None else None
        return cls(
            method=data["method"],
            active_orbitals=data.get("active_orbitals", []),
            n_active_orbitals=data.get("n_active_orbitals", 0),
            n_spin_orbitals=data.get("n_spin_orbitals", 0),
            n_active_electrons=data.get("n_active_electrons", 0.0),
            frozen_core_orbitals=data.get("frozen_core_orbitals", []),
            frozen_virtual_orbitals=data.get("frozen_virtual_orbitals", []),
            n_core_electrons=data.get("n_core_electrons", 0.0),
            energy_window_ev=energy_window_ev,
            fermi_energy_ev=data.get("fermi_energy_ev"),
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        """Human-readable description of the active space."""
        orbitals_str = (
            f"Bands {self.active_orbitals[0]}..{self.active_orbitals[-1]}"
            if self.active_orbitals and len(self.active_orbitals) > 1
            else str(self.active_orbitals)
        )
        return (
            f"ActiveSpace(method='{self.method}', "
            f"n_active_orbitals={self.n_active_orbitals}, "
            f"n_spin_orbitals={self.n_spin_orbitals}, "
            f"n_active_electrons={self.n_active_electrons:.1f}, "
            f"bands={orbitals_str})"
        )


class ActiveSpaceSelector(ABC):
    """Abstract base class for deterministic active space selectors."""

    @abstractmethod
    def select(self, state: QEElectronicState | QERunResult) -> ActiveSpace:
        """Construct active space from electronic structure state."""
        ...


def _resolve_electronic_state(source: QEElectronicState | QERunResult) -> QEElectronicState:
    """Extract QEElectronicState from either QEElectronicState or QERunResult."""
    if isinstance(source, QERunResult):
        return source.electronic
    if isinstance(source, QEElectronicState):
        return source
    raise TypeError(f"Expected QEElectronicState or QERunResult, got {type(source).__name__}")


class EnergyWindowSelector(ActiveSpaceSelector):
    """Select active orbitals based on an energy window around Fermi level or absolute energy.

    Parameters
    ----------
    emin_ev : float
        Lower bound of the energy window in eV.
    emax_ev : float
        Upper bound of the energy window in eV.
    relative_to_fermi : bool, optional
        If True (default), emin_ev and emax_ev are interpreted as offsets relative to E_Fermi.
        If False, emin_ev and emax_ev are treated as absolute band energies in eV.
    kpoint_mode : str, optional
        How to handle k-points ('gamma' or 'any' or 'average', default: 'gamma').
    """

    def __init__(
        self,
        emin_ev: float = -3.0,
        emax_ev: float = 3.0,
        relative_to_fermi: bool = True,
        kpoint_mode: str = "gamma",
    ) -> None:
        if emin_ev > emax_ev:
            raise ValueError(f"emin_ev ({emin_ev}) must be <= emax_ev ({emax_ev})")
        self.emin_ev = emin_ev
        self.emax_ev = emax_ev
        self.relative_to_fermi = relative_to_fermi
        self.kpoint_mode = kpoint_mode

    def select(self, state: QEElectronicState | QERunResult) -> ActiveSpace:
        el = _resolve_electronic_state(state)

        if not el.eigenvalues_ev:
            raise ValueError("No eigenvalues available in electronic state to select active space.")

        # Reference energy
        e_ref = 0.0
        if self.relative_to_fermi:
            if el.fermi_energy_ev is not None:
                e_ref = el.fermi_energy_ev
            elif el.highest_occupied_ev is not None:
                e_ref = el.highest_occupied_ev
            else:
                raise ValueError("No Fermi or HOMO energy found to compute relative energy window.")

        target_min = e_ref + self.emin_ev if self.relative_to_fermi else self.emin_ev
        target_max = e_ref + self.emax_ev if self.relative_to_fermi else self.emax_ev

        n_bands = el.n_bands or len(el.eigenvalues_ev[0])
        active_indices: list[int] = []

        # Check each band
        for b in range(n_bands):
            band_energies = [k_eigs[b] for k_eigs in el.eigenvalues_ev if b < len(k_eigs)]
            if not band_energies:
                continue

            if self.kpoint_mode == "gamma":
                eval_energy = band_energies[0]
            elif self.kpoint_mode == "average":
                eval_energy = sum(band_energies) / len(band_energies)
            else:  # "any"
                # If any k-point energy falls inside the window
                in_window = any(target_min <= e <= target_max for e in band_energies)
                if in_window:
                    active_indices.append(b)
                continue

            if target_min <= eval_energy <= target_max:
                active_indices.append(b)

        if not active_indices:
            raise ValueError(
                f"No bands found within energy window [{target_min:.3f}, {target_max:.3f}] eV."
            )

        # Core and virtual orbitals
        min_active = min(active_indices)
        max_active = max(active_indices)
        frozen_core = [b for b in range(min_active)]
        frozen_virtual = [b for b in range(max_active + 1, n_bands)]

        # Calculate electrons
        n_active_elec = 0.0
        n_core_elec = 0.0

        if el.occupations:
            # Average occupations across k-points
            for b in active_indices:
                b_occs = [k_occs[b] for k_occs in el.occupations if b < len(k_occs)]
                if b_occs:
                    n_active_elec += sum(b_occs) / len(b_occs)

            for b in frozen_core:
                b_occs = [k_occs[b] for k_occs in el.occupations if b < len(k_occs)]
                if b_occs:
                    n_core_elec += sum(b_occs) / len(b_occs)
        else:
            # Estimate from standard non-spin-polarized filling if unavailable
            if el.n_electrons is not None:
                # 2 electrons per band up to n_electrons
                for b in active_indices:
                    occ = max(0.0, min(2.0, el.n_electrons - 2.0 * b))
                    n_active_elec += occ
                for b in frozen_core:
                    occ = max(0.0, min(2.0, el.n_electrons - 2.0 * b))
                    n_core_elec += occ

        fermi_ref = el.fermi_energy_ev if el.fermi_energy_ev is not None else el.highest_occupied_ev

        return ActiveSpace(
            method="energy_window",
            active_orbitals=active_indices,
            n_active_orbitals=len(active_indices),
            n_spin_orbitals=len(active_indices) * 2,
            n_active_electrons=round(n_active_elec, 6),
            frozen_core_orbitals=frozen_core,
            frozen_virtual_orbitals=frozen_virtual,
            n_core_electrons=round(n_core_elec, 6),
            energy_window_ev=(target_min, target_max),
            fermi_energy_ev=fermi_ref,
            metadata={
                "emin_rel_ev": self.emin_ev if self.relative_to_fermi else None,
                "emax_rel_ev": self.emax_ev if self.relative_to_fermi else None,
                "relative_to_fermi": self.relative_to_fermi,
                "kpoint_mode": self.kpoint_mode,
            },
        )


class BandIndexSelector(ActiveSpaceSelector):
    """Select active space directly by specifying band indices or range.

    Parameters
    ----------
    band_indices : list of int, optional
        Explicit 0-indexed band indices.
    band_start : int, optional
        Start band index (0-indexed, inclusive).
    band_end : int, optional
        End band index (0-indexed, inclusive).
    """

    def __init__(
        self,
        band_indices: list[int] | None = None,
        band_start: int | None = None,
        band_end: int | None = None,
    ) -> None:
        if band_indices is not None:
            self.indices = sorted(set(band_indices))
        elif band_start is not None and band_end is not None:
            if band_start > band_end:
                raise ValueError(f"band_start ({band_start}) must be <= band_end ({band_end})")
            self.indices = list(range(band_start, band_end + 1))
        else:
            raise ValueError("Must specify either band_indices or both band_start and band_end.")

    def select(self, state: QEElectronicState | QERunResult) -> ActiveSpace:
        el = _resolve_electronic_state(state)
        n_bands = el.n_bands or (len(el.eigenvalues_ev[0]) if el.eigenvalues_ev else max(self.indices) + 1)

        for b in self.indices:
            if b < 0 or b >= n_bands:
                raise IndexError(f"Band index {b} out of range [0, {n_bands - 1}].")

        min_active = min(self.indices)
        max_active = max(self.indices)
        frozen_core = [b for b in range(min_active)]
        frozen_virtual = [b for b in range(max_active + 1, n_bands)]

        n_active_elec = 0.0
        n_core_elec = 0.0

        if el.occupations:
            for b in self.indices:
                b_occs = [k_occs[b] for k_occs in el.occupations if b < len(k_occs)]
                if b_occs:
                    n_active_elec += sum(b_occs) / len(b_occs)
            for b in frozen_core:
                b_occs = [k_occs[b] for k_occs in el.occupations if b < len(k_occs)]
                if b_occs:
                    n_core_elec += sum(b_occs) / len(b_occs)
        elif el.n_electrons is not None:
            for b in self.indices:
                occ = max(0.0, min(2.0, el.n_electrons - 2.0 * b))
                n_active_elec += occ
            for b in frozen_core:
                occ = max(0.0, min(2.0, el.n_electrons - 2.0 * b))
                n_core_elec += occ

        fermi_ref = el.fermi_energy_ev if el.fermi_energy_ev is not None else el.highest_occupied_ev

        return ActiveSpace(
            method="band_index",
            active_orbitals=self.indices,
            n_active_orbitals=len(self.indices),
            n_spin_orbitals=len(self.indices) * 2,
            n_active_electrons=round(n_active_elec, 6),
            frozen_core_orbitals=frozen_core,
            frozen_virtual_orbitals=frozen_virtual,
            n_core_electrons=round(n_core_elec, 6),
            fermi_energy_ev=fermi_ref,
            metadata={"specified_indices": self.indices},
        )


class OccupationSelector(ActiveSpaceSelector):
    """Select active space by selecting partially occupied bands (e.g. for metals/correlated systems).

    Parameters
    ----------
    min_occ : float, optional
        Minimum occupation threshold (default: 0.01).
    max_occ : float, optional
        Maximum occupation threshold (default: 1.99).
    """

    def __init__(self, min_occ: float = 0.01, max_occ: float = 1.99) -> None:
        self.min_occ = min_occ
        self.max_occ = max_occ

    def select(self, state: QEElectronicState | QERunResult) -> ActiveSpace:
        el = _resolve_electronic_state(state)
        if not el.occupations:
            raise ValueError("No occupations available in electronic state for OccupationSelector.")

        n_bands = el.n_bands or len(el.occupations[0])
        active_indices: list[int] = []

        for b in range(n_bands):
            b_occs = [k_occs[b] for k_occs in el.occupations if b < len(k_occs)]
            if not b_occs:
                continue
            avg_occ = sum(b_occs) / len(b_occs)
            if self.min_occ <= avg_occ <= self.max_occ:
                active_indices.append(b)

        if not active_indices:
            raise ValueError(
                f"No partially occupied bands found within occupation range [{self.min_occ}, {self.max_occ}]."
            )

        min_active = min(active_indices)
        max_active = max(active_indices)
        frozen_core = [b for b in range(min_active)]
        frozen_virtual = [b for b in range(max_active + 1, n_bands)]

        n_active_elec = 0.0
        n_core_elec = 0.0
        for b in active_indices:
            b_occs = [k_occs[b] for k_occs in el.occupations if b < len(k_occs)]
            if b_occs:
                n_active_elec += sum(b_occs) / len(b_occs)

        for b in frozen_core:
            b_occs = [k_occs[b] for k_occs in el.occupations if b < len(k_occs)]
            if b_occs:
                n_core_elec += sum(b_occs) / len(b_occs)

        fermi_ref = el.fermi_energy_ev if el.fermi_energy_ev is not None else el.highest_occupied_ev

        return ActiveSpace(
            method="occupation",
            active_orbitals=active_indices,
            n_active_orbitals=len(active_indices),
            n_spin_orbitals=len(active_indices) * 2,
            n_active_electrons=round(n_active_elec, 6),
            frozen_core_orbitals=frozen_core,
            frozen_virtual_orbitals=frozen_virtual,
            n_core_electrons=round(n_core_elec, 6),
            fermi_energy_ev=fermi_ref,
            metadata={"min_occ": self.min_occ, "max_occ": self.max_occ},
        )


class ExplicitOrbitalSelector(ActiveSpaceSelector):
    """Explicit orbital assignment with user-specified electron count."""

    def __init__(
        self,
        active_orbitals: list[int],
        n_active_electrons: float,
        frozen_core_orbitals: list[int] | None = None,
        frozen_virtual_orbitals: list[int] | None = None,
    ) -> None:
        self.active_orbitals = sorted(set(active_orbitals))
        self.n_active_electrons = n_active_electrons
        self.frozen_core_orbitals = sorted(set(frozen_core_orbitals or []))
        self.frozen_virtual_orbitals = sorted(set(frozen_virtual_orbitals or []))

    def select(self, state: QEElectronicState | QERunResult) -> ActiveSpace:
        el = _resolve_electronic_state(state)
        return ActiveSpace(
            method="explicit",
            active_orbitals=self.active_orbitals,
            n_active_orbitals=len(self.active_orbitals),
            n_spin_orbitals=len(self.active_orbitals) * 2,
            n_active_electrons=self.n_active_electrons,
            frozen_core_orbitals=self.frozen_core_orbitals,
            frozen_virtual_orbitals=self.frozen_virtual_orbitals,
            fermi_energy_ev=el.fermi_energy_ev,
            metadata={"manual_override": True},
        )


def create_active_space_selector(method: str = "energy_window", **kwargs: Any) -> ActiveSpaceSelector:
    """Factory function for creating an active space selector by strategy name."""
    m = method.lower()
    if m in ("energy_window", "window", "energy"):
        return EnergyWindowSelector(**kwargs)
    if m in ("band_index", "bands", "band"):
        return BandIndexSelector(**kwargs)
    if m in ("occupation", "occ"):
        return OccupationSelector(**kwargs)
    if m in ("explicit", "manual"):
        return ExplicitOrbitalSelector(**kwargs)
    raise ValueError(f"Unknown active space selector method: '{method}'. Choose from: 'energy_window', 'band_index', 'occupation', 'explicit'.")


def select_active_space(
    source: QEElectronicState | QERunResult,
    method: str = "energy_window",
    **kwargs: Any,
) -> ActiveSpace:
    """Convenience function to select an active space directly from a state or result."""
    selector = create_active_space_selector(method=method, **kwargs)
    return selector.select(source)
