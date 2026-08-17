"""Active-space selection for DFT-to-correlated-solver handoff.

The selectors in this module operate on QE *band indices*.  For periodic
calculations that is a diagnostic/model-selection layer, not a claim that one
Bloch band is already one localized correlated orbital.  A physical finite
Hamiltonian should normally be constructed after Wannier/downfolding (or from
explicit one-/two-electron integrals).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qeanalyzer.models import QEElectronicState, QERunResult


@dataclass
class ActiveSpace:
    method: str
    active_orbitals: list[int] = field(default_factory=list)
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
        self.active_orbitals = sorted(set(self.active_orbitals))
        self.frozen_core_orbitals = sorted(set(self.frozen_core_orbitals))
        self.frozen_virtual_orbitals = sorted(set(self.frozen_virtual_orbitals))
        if not self.n_active_orbitals:
            self.n_active_orbitals = len(self.active_orbitals)
        if not self.n_spin_orbitals:
            self.n_spin_orbitals = 2 * self.n_active_orbitals

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "ActiveSpace":
        window = data.get("energy_window_ev")
        return cls(
            method=data["method"],
            active_orbitals=data.get("active_orbitals", []),
            n_active_orbitals=data.get("n_active_orbitals", 0),
            n_spin_orbitals=data.get("n_spin_orbitals", 0),
            n_active_electrons=data.get("n_active_electrons", 0.0),
            frozen_core_orbitals=data.get("frozen_core_orbitals", []),
            frozen_virtual_orbitals=data.get("frozen_virtual_orbitals", []),
            n_core_electrons=data.get("n_core_electrons", 0.0),
            energy_window_ev=tuple(window) if window is not None else None,
            fermi_energy_ev=data.get("fermi_energy_ev"),
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        if len(self.active_orbitals) > 1:
            orbitals = f"Bands {self.active_orbitals[0]}..{self.active_orbitals[-1]}"
        else:
            orbitals = str(self.active_orbitals)
        return (
            f"ActiveSpace(method={self.method!r}, n_active_orbitals={self.n_active_orbitals}, "
            f"n_spin_orbitals={self.n_spin_orbitals}, "
            f"n_active_electrons={self.n_active_electrons:.6g}, bands={orbitals})"
        )


class ActiveSpaceSelector(ABC):
    @abstractmethod
    def select(self, state: QEElectronicState | QERunResult) -> ActiveSpace:
        ...


def _resolve_electronic_state(source: QEElectronicState | QERunResult) -> QEElectronicState:
    if isinstance(source, QERunResult):
        return source.electronic
    if isinstance(source, QEElectronicState):
        return source
    raise TypeError(f"Expected QEElectronicState or QERunResult, got {type(source).__name__}")


def _require_restricted_band_semantics(el: QEElectronicState) -> None:
    """Reject spin representations that cannot be mapped by 2*b + spin."""
    if el.lsda or el.noncolin or el.spinorbit:
        modes = []
        if el.lsda:
            modes.append("LSDA/spin-polarized")
        if el.noncolin:
            modes.append("noncollinear")
        if el.spinorbit:
            modes.append("spin-orbit")
        raise NotImplementedError(
            "Restricted band-index active spaces assume one spatial band with alpha/beta partners. "
            f"This QE result is {', '.join(modes)}; use an explicit spinor/Wannier mapping instead."
        )


def _weights(el: QEElectronicState, n: int) -> list[float]:
    if n <= 0:
        return []
    weights = el.normalized_kpoint_weights()
    if not weights:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError("k-point weights are inconsistent with the k-resolved data")
    return weights


def _weighted_band_average(rows: list[list[float]], band: int, weights: list[float]) -> float | None:
    pairs = [(row[band], weights[k]) for k, row in enumerate(rows) if band < len(row)]
    if not pairs:
        return None
    denom = sum(w for _, w in pairs)
    if denom <= 0.0:
        return None
    return sum(value * w for value, w in pairs) / denom


def _electron_partition(el: QEElectronicState, active: list[int], frozen_core: list[int]) -> tuple[float, float]:
    if el.occupations:
        weights = _weights(el, len(el.occupations))
        active_e = sum((_weighted_band_average(el.occupations, b, weights) or 0.0) for b in active)
        core_e = sum((_weighted_band_average(el.occupations, b, weights) or 0.0) for b in frozen_core)
        return active_e, core_e
    if el.n_electrons is None:
        return 0.0, 0.0
    def fill(b: int) -> float:
        return max(0.0, min(2.0, float(el.n_electrons) - 2.0 * b))
    return sum(fill(b) for b in active), sum(fill(b) for b in frozen_core)


def _build_space(el: QEElectronicState, *, method: str, active: list[int], metadata: dict[str, Any],
                 energy_window_ev: tuple[float, float] | None = None) -> ActiveSpace:
    if not active:
        raise ValueError("active-space selection produced no bands")
    n_bands = el.n_bands or max(active) + 1
    minimum, maximum = min(active), max(active)
    frozen_core = list(range(minimum))
    frozen_virtual = list(range(maximum + 1, n_bands))
    active_e, core_e = _electron_partition(el, active, frozen_core)
    fermi = el.fermi_energy_ev if el.fermi_energy_ev is not None else el.highest_occupied_ev
    return ActiveSpace(
        method=method,
        active_orbitals=active,
        n_active_orbitals=len(active),
        n_spin_orbitals=2 * len(active),
        n_active_electrons=round(active_e, 10),
        frozen_core_orbitals=frozen_core,
        frozen_virtual_orbitals=frozen_virtual,
        n_core_electrons=round(core_e, 10),
        energy_window_ev=energy_window_ev,
        fermi_energy_ev=fermi,
        metadata={
            "representation": "periodic_band_indices",
            "physical_hamiltonian_ready": False,
            **metadata,
        },
    )


class EnergyWindowSelector(ActiveSpaceSelector):
    def __init__(self, emin_ev: float = -3.0, emax_ev: float = 3.0,
                 relative_to_fermi: bool = True, kpoint_mode: str = "gamma") -> None:
        if emin_ev > emax_ev:
            raise ValueError(f"emin_ev ({emin_ev}) must be <= emax_ev ({emax_ev})")
        if kpoint_mode not in {"gamma", "average", "any"}:
            raise ValueError("kpoint_mode must be 'gamma', 'average', or 'any'")
        self.emin_ev = emin_ev
        self.emax_ev = emax_ev
        self.relative_to_fermi = relative_to_fermi
        self.kpoint_mode = kpoint_mode

    def select(self, state: QEElectronicState | QERunResult) -> ActiveSpace:
        el = _resolve_electronic_state(state)
        _require_restricted_band_semantics(el)
        if not el.eigenvalues_ev:
            raise ValueError("No eigenvalues available to select an active space")
        reference = 0.0
        if self.relative_to_fermi:
            if el.fermi_energy_ev is not None:
                reference = el.fermi_energy_ev
            elif el.highest_occupied_ev is not None:
                reference = el.highest_occupied_ev
            else:
                raise ValueError("No Fermi/HOMO energy is available for a relative window")
        low = reference + self.emin_ev if self.relative_to_fermi else self.emin_ev
        high = reference + self.emax_ev if self.relative_to_fermi else self.emax_ev
        n_bands = el.n_bands or len(el.eigenvalues_ev[0])
        weights = _weights(el, len(el.eigenvalues_ev))
        active: list[int] = []
        for band in range(n_bands):
            vals = [row[band] for row in el.eigenvalues_ev if band < len(row)]
            if not vals:
                continue
            if self.kpoint_mode == "any":
                include = any(low <= value <= high for value in vals)
            elif self.kpoint_mode == "average":
                avg = _weighted_band_average(el.eigenvalues_ev, band, weights)
                include = avg is not None and low <= avg <= high
            else:
                include = low <= vals[0] <= high
            if include:
                active.append(band)
        if not active:
            raise ValueError(f"No bands found within energy window [{low:.3f}, {high:.3f}] eV")
        return _build_space(
            el, method="energy_window", active=active, energy_window_ev=(low, high),
            metadata={
                "emin_rel_ev": self.emin_ev if self.relative_to_fermi else None,
                "emax_rel_ev": self.emax_ev if self.relative_to_fermi else None,
                "relative_to_fermi": self.relative_to_fermi,
                "kpoint_mode": self.kpoint_mode,
                "occupation_average": "kpoint_weighted",
            },
        )


class BandIndexSelector(ActiveSpaceSelector):
    def __init__(self, band_indices: list[int] | None = None,
                 band_start: int | None = None, band_end: int | None = None) -> None:
        if band_indices is not None:
            self.indices = sorted(set(band_indices))
        elif band_start is not None and band_end is not None:
            if band_start > band_end:
                raise ValueError(f"band_start ({band_start}) must be <= band_end ({band_end})")
            self.indices = list(range(band_start, band_end + 1))
        else:
            raise ValueError("Specify band_indices or both band_start and band_end")
        if not self.indices:
            raise ValueError("band selection cannot be empty")

    def select(self, state: QEElectronicState | QERunResult) -> ActiveSpace:
        el = _resolve_electronic_state(state)
        _require_restricted_band_semantics(el)
        n_bands = el.n_bands or (len(el.eigenvalues_ev[0]) if el.eigenvalues_ev else max(self.indices) + 1)
        for band in self.indices:
            if band < 0 or band >= n_bands:
                raise IndexError(f"Band index {band} out of range [0, {n_bands - 1}]")
        return _build_space(
            el, method="band_index", active=list(self.indices),
            metadata={"specified_indices": list(self.indices), "occupation_average": "kpoint_weighted"},
        )


class OccupationSelector(ActiveSpaceSelector):
    def __init__(self, min_occ: float = 0.01, max_occ: float = 1.99) -> None:
        if min_occ > max_occ:
            raise ValueError("min_occ must be <= max_occ")
        self.min_occ, self.max_occ = min_occ, max_occ

    def select(self, state: QEElectronicState | QERunResult) -> ActiveSpace:
        el = _resolve_electronic_state(state)
        _require_restricted_band_semantics(el)
        if not el.occupations:
            raise ValueError("No occupations available for OccupationSelector")
        n_bands = el.n_bands or len(el.occupations[0])
        weights = _weights(el, len(el.occupations))
        active = []
        for band in range(n_bands):
            avg = _weighted_band_average(el.occupations, band, weights)
            if avg is not None and self.min_occ <= avg <= self.max_occ:
                active.append(band)
        if not active:
            raise ValueError(
                f"No bands found within occupation range [{self.min_occ}, {self.max_occ}]"
            )
        return _build_space(
            el, method="occupation", active=active,
            metadata={
                "min_occ": self.min_occ,
                "max_occ": self.max_occ,
                "occupation_average": "kpoint_weighted",
            },
        )


class ExplicitOrbitalSelector(ActiveSpaceSelector):
    def __init__(self, active_orbitals: list[int], n_active_electrons: float,
                 frozen_core_orbitals: list[int] | None = None,
                 frozen_virtual_orbitals: list[int] | None = None) -> None:
        if not active_orbitals:
            raise ValueError("active_orbitals cannot be empty")
        self.active_orbitals = sorted(set(active_orbitals))
        self.n_active_electrons = float(n_active_electrons)
        self.frozen_core = sorted(set(frozen_core_orbitals or []))
        self.frozen_virtual = sorted(set(frozen_virtual_orbitals or []))

    def select(self, state: QEElectronicState | QERunResult) -> ActiveSpace:
        el = _resolve_electronic_state(state)
        return ActiveSpace(
            method="explicit",
            active_orbitals=self.active_orbitals,
            n_active_orbitals=len(self.active_orbitals),
            n_spin_orbitals=2 * len(self.active_orbitals),
            n_active_electrons=self.n_active_electrons,
            frozen_core_orbitals=self.frozen_core,
            frozen_virtual_orbitals=self.frozen_virtual,
            n_core_electrons=2.0 * len(self.frozen_core),
            fermi_energy_ev=el.fermi_energy_ev,
            metadata={"manual_override": True, "representation": "explicit_orbital_indices"},
        )


def create_active_space_selector(method: str, **kwargs: Any) -> ActiveSpaceSelector:
    key = method.lower().replace("-", "_")
    if key in {"energy_window", "energy"}:
        return EnergyWindowSelector(**kwargs)
    if key in {"band_index", "band", "bands"}:
        return BandIndexSelector(**kwargs)
    if key in {"occupation", "occupations", "occ"}:
        return OccupationSelector(**kwargs)
    if key in {"explicit", "manual"}:
        return ExplicitOrbitalSelector(**kwargs)
    raise ValueError(f"Unknown active-space selection method {method!r}")


def select_active_space(state: QEElectronicState | QERunResult, method: str = "energy_window", **kwargs: Any) -> ActiveSpace:
    return create_active_space_selector(method, **kwargs).select(state)
