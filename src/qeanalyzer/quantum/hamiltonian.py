"""Canonical MaterialHamiltonian representation for downfolded and active-space electronic models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from qeanalyzer.models import QEElectronicState, QERunResult
from qeanalyzer.quantum.active_space import ActiveSpace


@dataclass
class MaterialHamiltonian:
    """Canonical representation of an electronic many-body Hamiltonian.

    Supports both explicit 1-body (h1) and 2-body (h2) electronic integral tensors
    as well as parameterized tight-binding / Hubbard (t, U, V, J) lattice models.

    Convention:
        H = E_const
          + sum_{pq, sigma} h1_{pq} c_{p,sigma}^dagger c_{q,sigma}
          + 0.5 * sum_{pqrs, sigma, tau} h2_{pqrs} c_{p,sigma}^dagger c_{r,tau}^dagger c_{s,tau} c_{q,sigma}

    In chemists' notation:
        h2_{pqrs} = (pr | qs) = int phi_p^*(r1) phi_r^*(r2) 1/|r1-r2| phi_q(r1) phi_s(r2) dr1 dr2
    """

    n_orbitals: int  # Number of spatial orbitals
    n_electrons: float  # Number of electrons in the active space
    constant: float = 0.0  # Constant shift (e.g. core energy + nuclear repulsion) in eV / Ry
    spin: int = 0  # 2S (multiplicity - 1)
    energy_unit: str = "eV"  # "eV", "Ry", or "Hartree"

    # 1-body integrals: shape (n_orbitals, n_orbitals)
    h1: list[list[float]] = field(default_factory=list)

    # 2-body integrals: shape (n_orbitals, n_orbitals, n_orbitals, n_orbitals)
    h2: list[list[list[list[float]]]] = field(default_factory=list)

    # Optional model parameters
    hopping_t: dict[tuple[int, int], float] = field(default_factory=dict)
    onsite_u: list[float] = field(default_factory=list)
    intersite_v: dict[tuple[int, int], float] = field(default_factory=dict)
    exchange_j: dict[tuple[int, int], float] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize empty tensors if not explicitly provided."""
        if self.n_orbitals <= 0:
            raise ValueError(f"n_orbitals must be > 0, got {self.n_orbitals}")

        # Initialize h1 if empty
        if not self.h1:
            self.h1 = [[0.0 for _ in range(self.n_orbitals)] for _ in range(self.n_orbitals)]

        # Initialize h2 if empty
        if not self.h2:
            self.h2 = [
                [
                    [
                        [0.0 for _ in range(self.n_orbitals)]
                        for _ in range(self.n_orbitals)
                    ]
                    for _ in range(self.n_orbitals)
                ]
                for _ in range(self.n_orbitals)
            ]

        # Initialize onsite_u list if empty
        if not self.onsite_u:
            self.onsite_u = [0.0] * self.n_orbitals

    @property
    def n_spin_orbitals(self) -> int:
        """Total number of spin orbitals (2 * n_spatial_orbitals)."""
        return 2 * self.n_orbitals

    def is_hermitian(self, tolerance: float = 1e-7) -> bool:
        """Check Hermiticity of 1-body matrix and symmetry of 2-body tensor."""
        norb = self.n_orbitals

        # Check 1-body Hermiticity: h1[p][q] == h1[q][p]
        for p in range(norb):
            for q in range(norb):
                if abs(self.h1[p][q] - self.h1[q][p]) > tolerance:
                    return False

        # Check 2-body 8-fold real symmetry: (pq|rs) == (qp|rs) == (pq|sr) == (rs|pq)
        for p in range(norb):
            for q in range(norb):
                for r in range(norb):
                    for s in range(norb):
                        val = self.h2[p][q][r][s]
                        if abs(val - self.h2[q][p][r][s]) > tolerance:
                            return False
                        if abs(val - self.h2[p][q][s][r]) > tolerance:
                            return False
                        if abs(val - self.h2[r][s][p][q]) > tolerance:
                            return False

        return True

    def count_nonzero_integrals(self, tolerance: float = 1e-9) -> tuple[int, int]:
        """Return counts of non-zero elements in (h1, h2)."""
        norb = self.n_orbitals
        n1 = sum(1 for p in range(norb) for q in range(norb) if abs(self.h1[p][q]) > tolerance)
        n2 = sum(
            1 for p in range(norb) for q in range(norb) for r in range(norb) for s in range(norb)
            if abs(self.h2[p][q][r][s]) > tolerance
        )
        return n1, n2

    def to_spin_orbital_integrals(self) -> tuple[list[list[float]], list[list[list[list[float]]]]]:
        """Convert spatial integrals to spin-orbital basis (alpha: 2p, beta: 2p+1).

        Returns
        -------
        h1_spin : 2D list of shape (2*norb, 2*norb)
        h2_spin : 4D list of shape (2*norb, 2*norb, 2*norb, 2*norb)
            In physicists' anti-symmetrized convention: <pr||qs> = <pr|qs> - <pr|sq>
        """
        n_so = self.n_spin_orbitals
        norb = self.n_orbitals

        h1_so = [[0.0 for _ in range(n_so)] for _ in range(n_so)]
        h2_so = [[[[0.0 for _ in range(n_so)] for _ in range(n_so)] for _ in range(n_so)] for _ in range(n_so)]

        # 1-body spin conversion: delta_{sigma, sigma'} h1[p, q]
        for p in range(norb):
            for q in range(norb):
                val = self.h1[p][q]
                h1_so[2 * p][2 * q] = val          # alpha-alpha
                h1_so[2 * p + 1][2 * q + 1] = val  # beta-beta

        # 2-body spin conversion in anti-symmetrized physicists' notation <pr||qs>
        # <p_s1, r_s2 | q_s1, s_s2> = (pq|rs)
        for p in range(norb):
            for q in range(norb):
                for r in range(norb):
                    for s in range(norb):
                        chem_val = self.h2[p][q][r][s]  # (pq|rs)
                        if abs(chem_val) < 1e-12:
                            continue

                        # alpha-alpha / beta-beta pairs
                        for s1 in (0, 1):
                            for s2 in (0, 1):
                                p_idx = 2 * p + s1
                                r_idx = 2 * r + s2
                                q_idx = 2 * q + s1
                                s_idx = 2 * s + s2

                                # Direct term: <p_idx, r_idx | q_idx, s_idx>
                                h2_so[p_idx][r_idx][q_idx][s_idx] += chem_val

                                # Exchange term if same spin: - <p_idx, r_idx | s_idx, q_idx>
                                if s1 == s2:
                                    h2_so[p_idx][r_idx][s_idx][q_idx] -= chem_val

        return h1_so, h2_so

    def to_dict(self) -> dict[str, Any]:
        """Serialize MaterialHamiltonian to dictionary."""
        # Convert tuple keys to str for JSON compatibility
        hopping_dict = {f"{k[0]},{k[1]}": v for k, v in self.hopping_t.items()}
        intersite_dict = {f"{k[0]},{k[1]}": v for k, v in self.intersite_v.items()}
        exchange_dict = {f"{k[0]},{k[1]}": v for k, v in self.exchange_j.items()}

        return {
            "n_orbitals": self.n_orbitals,
            "n_spin_orbitals": self.n_spin_orbitals,
            "n_electrons": self.n_electrons,
            "constant": self.constant,
            "spin": self.spin,
            "energy_unit": self.energy_unit,
            "h1": self.h1,
            "h2": self.h2,
            "hopping_t": hopping_dict,
            "onsite_u": self.onsite_u,
            "intersite_v": intersite_dict,
            "exchange_j": exchange_dict,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaterialHamiltonian:
        """Construct MaterialHamiltonian from dictionary."""
        hopping_t = {}
        for k_str, val in data.get("hopping_t", {}).items():
            p, q = map(int, k_str.split(","))
            hopping_t[(p, q)] = float(val)

        intersite_v = {}
        for k_str, val in data.get("intersite_v", {}).items():
            p, q = map(int, k_str.split(","))
            intersite_v[(p, q)] = float(val)

        exchange_j = {}
        for k_str, val in data.get("exchange_j", {}).items():
            p, q = map(int, k_str.split(","))
            exchange_j[(p, q)] = float(val)

        return cls(
            n_orbitals=data["n_orbitals"],
            n_electrons=data.get("n_electrons", 0.0),
            constant=data.get("constant", 0.0),
            spin=data.get("spin", 0),
            energy_unit=data.get("energy_unit", "eV"),
            h1=data.get("h1", []),
            h2=data.get("h2", []),
            hopping_t=hopping_t,
            onsite_u=data.get("onsite_u", []),
            intersite_v=intersite_v,
            exchange_j=exchange_j,
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        """Human-readable overview of Hamiltonian parameters and size."""
        n1, n2 = self.count_nonzero_integrals()
        lines = [
            "MaterialHamiltonian Summary",
            "=" * 40,
            f"Spatial Orbitals  : {self.n_orbitals}",
            f"Spin Orbitals     : {self.n_spin_orbitals}",
            f"Active Electrons  : {self.n_electrons:.1f}",
            f"Constant Energy   : {self.constant:.6f} {self.energy_unit}",
            f"1-Body Integrals  : {n1} non-zero elements",
            f"2-Body Integrals  : {n2} non-zero elements",
            f"Hermitian Valid   : {self.is_hermitian()}",
        ]
        if any(abs(u) > 1e-6 for u in self.onsite_u):
            u_str = ", ".join(f"{u:.2f}" for u in self.onsite_u[:4])
            if len(self.onsite_u) > 4:
                u_str += ", ..."
            lines.append(f"Onsite Hubbard U  : [{u_str}] {self.energy_unit}")
        if self.hopping_t:
            lines.append(f"Hopping Terms (t) : {len(self.hopping_t)} couplings")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Factory & Model Builders
# -----------------------------------------------------------------------------

def build_hubbard_hamiltonian(
    n_orbitals: int,
    n_electrons: float,
    hopping_t: dict[tuple[int, int], float] | list[list[float]] | None = None,
    onsite_u: float | list[float] | None = None,
    intersite_v: dict[tuple[int, int], float] | None = None,
    constant: float = 0.0,
    energy_unit: str = "eV",
) -> MaterialHamiltonian:
    """Build a tight-binding Hubbard model MaterialHamiltonian with full 1-body and 2-body integrals.

    Parameters
    ----------
    n_orbitals : int
        Number of spatial sites / orbitals.
    n_electrons : float
        Total number of electrons.
    hopping_t : dict or 2D list, optional
        Hopping amplitudes t_ij between sites (e.g. {(0, 1): -1.0, (1, 0): -1.0}).
    onsite_u : float or list of float, optional
        Onsite Coulomb repulsion U_i (eV).
    intersite_v : dict, optional
        Intersite Coulomb repulsion V_ij between sites i and j.
    constant : float, optional
        Scalar energy offset.
    energy_unit : str, optional
        Unit of energy (default: 'eV').

    Returns
    -------
    MaterialHamiltonian
        Populated Hamiltonian with h1 and h2 tensors.
    """
    h1 = [[0.0 for _ in range(n_orbitals)] for _ in range(n_orbitals)]
    h2 = [[[[0.0 for _ in range(n_orbitals)] for _ in range(n_orbitals)] for _ in range(n_orbitals)] for _ in range(n_orbitals)]

    # Parse hopping_t
    hop_dict: dict[tuple[int, int], float] = {}
    if hopping_t is not None:
        if isinstance(hopping_t, dict):
            hop_dict = dict(hopping_t)
        elif isinstance(hopping_t, list):
            for i in range(n_orbitals):
                for j in range(n_orbitals):
                    val = hopping_t[i][j]
                    if abs(val) > 1e-9:
                        hop_dict[(i, j)] = val

    # Fill 1-body matrix h1
    for (i, j), t_val in hop_dict.items():
        if 0 <= i < n_orbitals and 0 <= j < n_orbitals:
            h1[i][j] = -t_val if t_val > 0 and (i != j) else t_val

    # Parse onsite_u
    u_list = [0.0] * n_orbitals
    if onsite_u is not None:
        if isinstance(onsite_u, (int, float)):
            u_list = [float(onsite_u)] * n_orbitals
        elif isinstance(onsite_u, list):
            u_list = [float(u) for u in onsite_u[:n_orbitals]] + [0.0] * max(0, n_orbitals - len(onsite_u))

    # Fill 2-body onsite repulsion: (ii|ii) = U_i
    for i in range(n_orbitals):
        u_val = u_list[i]
        if abs(u_val) > 1e-9:
            h2[i][i][i][i] = u_val

    # Parse intersite_v
    v_dict: dict[tuple[int, int], float] = {}
    if intersite_v is not None:
        v_dict = dict(intersite_v)
        for (i, j), v_val in v_dict.items():
            if 0 <= i < n_orbitals and 0 <= j < n_orbitals and abs(v_val) > 1e-9:
                # Intersite density-density (ij|ij) = (ji|ji) = V_ij
                h2[i][i][j][j] = v_val
                h2[j][j][i][i] = v_val

    return MaterialHamiltonian(
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
        constant=constant,
        energy_unit=energy_unit,
        h1=h1,
        h2=h2,
        hopping_t=hop_dict,
        onsite_u=u_list,
        intersite_v=v_dict,
        metadata={"model_type": "hubbard"},
    )


def build_active_space_hamiltonian(
    state: QEElectronicState | QERunResult,
    active_space: ActiveSpace,
    onsite_u_ev: float = 0.0,
    intersite_v_ev: float = 0.0,
) -> MaterialHamiltonian:
    """Build a MaterialHamiltonian for an active space directly from DFT eigenvalues.

    Parameters
    ----------
    state : QEElectronicState or QERunResult
        The source electronic structure state containing band eigenvalues and occupations.
    active_space : ActiveSpace
        The selected active space partition.
    onsite_u_ev : float, optional
        Effective Hubbard U interaction (eV) to apply to active frontier orbitals.
    intersite_v_ev : float, optional
        Effective nearest-neighbor intersite Coulomb interaction (eV).

    Returns
    -------
    MaterialHamiltonian
        Hamiltonian spanning the active space orbitals.
    """
    if isinstance(state, QERunResult):
        el = state.electronic
        tot_energy = state.electronic.total_energy_ev or (
            state.electronic.total_energy_ry * 13.6056980659 if state.electronic.total_energy_ry else 0.0
        )
    else:
        el = state
        tot_energy = el.total_energy_ev or (el.total_energy_ry * 13.6056980659 if el.total_energy_ry else 0.0)

    norb = active_space.n_active_orbitals
    if norb == 0:
        raise ValueError("Cannot construct Hamiltonian from an empty active space.")

    h1 = [[0.0 for _ in range(norb)] for _ in range(norb)]
    h2 = [[[[0.0 for _ in range(norb)] for _ in range(norb)] for _ in range(norb)] for _ in range(norb)]

    # 1-body terms: diagonal Kohn-Sham orbital energies (averaged over k-points)
    active_band_indices = active_space.active_orbitals
    for loc_i, band_idx in enumerate(active_band_indices):
        if el.eigenvalues_ev:
            eigs = [k_eigs[band_idx] for k_eigs in el.eigenvalues_ev if band_idx < len(k_eigs)]
            if eigs:
                h1[loc_i][loc_i] = sum(eigs) / len(eigs)

    # 2-body interaction terms
    if abs(onsite_u_ev) > 1e-9:
        for i in range(norb):
            h2[i][i][i][i] = onsite_u_ev

    if abs(intersite_v_ev) > 1e-9:
        for i in range(norb - 1):
            h2[i][i][i + 1][i + 1] = intersite_v_ev
            h2[i + 1][i + 1][i][i] = intersite_v_ev

    # Core electron contribution to constant shift
    core_shift = 0.0
    if active_space.frozen_core_orbitals and el.eigenvalues_ev:
        for band_idx in active_space.frozen_core_orbitals:
            eigs = [k_eigs[band_idx] for k_eigs in el.eigenvalues_ev if band_idx < len(k_eigs)]
            if eigs:
                core_shift += 2.0 * (sum(eigs) / len(eigs))

    return MaterialHamiltonian(
        n_orbitals=norb,
        n_electrons=active_space.n_active_electrons,
        constant=round(core_shift, 6),
        energy_unit="eV",
        h1=h1,
        h2=h2,
        onsite_u=[onsite_u_ev] * norb,
        metadata={
            "model_type": "dft_active_space",
            "active_space_method": active_space.method,
            "active_bands": active_band_indices,
            "dft_total_energy_ev": tot_energy,
        },
    )
