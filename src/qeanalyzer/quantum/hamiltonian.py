"""Canonical finite active-space Hamiltonians and explicitly labelled model builders."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

from qeanalyzer.models import QEElectronicState, QERunResult
from qeanalyzer.quantum.active_space import ActiveSpace
from qeanalyzer.quantum.units import normalize_energy_unit


@dataclass
class MaterialHamiltonian:
    """Finite restricted spatial-orbital Hamiltonian.

    The integral convention is fixed throughout qeanalyzer and at FCIDUMP:

    ``h2[p][q][r][s] == (pq|rs)`` (chemist notation), and

    ``H2 = 1/2 sum_pqrs sum_st (pq|rs)
          a†_(p,s) a†_(r,t) a_(s,t) a_(q,s)``.

    Arrays are real.  General complex/spinor/unrestricted Hamiltonians require a
    different model type and are intentionally rejected by the restricted
    adapters rather than silently coerced.
    """

    n_orbitals: int
    n_electrons: float
    constant: float = 0.0
    spin: int = 0  # MS2 = N_alpha - N_beta
    energy_unit: str = "eV"
    h1: list[list[float]] = field(default_factory=list)
    h2: list[list[list[list[float]]]] = field(default_factory=list)
    hopping_t: dict[tuple[int, int], float] = field(default_factory=dict)
    onsite_u: list[float] = field(default_factory=list)
    intersite_v: dict[tuple[int, int], float] = field(default_factory=dict)
    exchange_j: dict[tuple[int, int], float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_orbitals <= 0:
            raise ValueError("n_orbitals must be positive")
        self.energy_unit = normalize_energy_unit(self.energy_unit)
        n = self.n_orbitals
        if not self.h1:
            self.h1 = [[0.0] * n for _ in range(n)]
        if not self.h2:
            self.h2 = [[[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
        if len(self.h1) != n or any(len(row) != n for row in self.h1):
            raise ValueError("h1 must have shape (n_orbitals, n_orbitals)")
        if len(self.h2) != n:
            raise ValueError("h2 must have shape (n_orbitals,)*4")
        for a in self.h2:
            if len(a) != n:
                raise ValueError("h2 must have shape (n_orbitals,)*4")
            for b in a:
                if len(b) != n or any(len(c) != n for c in b):
                    raise ValueError("h2 must have shape (n_orbitals,)*4")
        if not self.onsite_u:
            self.onsite_u = [0.0] * n

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * self.n_orbitals

    def is_hermitian(self, tolerance: float = 1e-7) -> bool:
        n = self.n_orbitals
        for p in range(n):
            for q in range(n):
                if abs(self.h1[p][q] - self.h1[q][p]) > tolerance:
                    return False
        # Real spatial ERIs: (pq|rs)=(qp|rs)=(pq|sr)=(rs|pq)
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        value = self.h2[p][q][r][s]
                        if abs(value - self.h2[q][p][r][s]) > tolerance:
                            return False
                        if abs(value - self.h2[p][q][s][r]) > tolerance:
                            return False
                        if abs(value - self.h2[r][s][p][q]) > tolerance:
                            return False
        return True

    def count_nonzero_integrals(self, tolerance: float = 1e-9) -> tuple[int, int]:
        n = self.n_orbitals
        one = sum(abs(self.h1[p][q]) > tolerance for p in range(n) for q in range(n))
        two = sum(
            abs(self.h2[p][q][r][s]) > tolerance
            for p in range(n) for q in range(n) for r in range(n) for s in range(n)
        )
        return int(one), int(two)

    def to_spin_orbital_integrals(self) -> tuple[list[list[float]], list[list[list[list[float]]]]]:
        """Return h1 and antisymmetrized two-electron spin-orbital tensors.

        The two-body return uses indices ``[P][R][Q][S]`` for
        ``<P R || Q S>`` and interleaved spin orbitals ``2*p=alpha``,
        ``2*p+1=beta``.  This helper is kept for compatibility; the canonical
        stored tensor remains spatial chemist ``(pq|rs)``.
        """
        n = self.n_orbitals
        nso = 2 * n
        one = [[0.0 for _ in range(nso)] for _ in range(nso)]
        two = [[[[0.0 for _ in range(nso)] for _ in range(nso)] for _ in range(nso)] for _ in range(nso)]
        for p in range(n):
            for q in range(n):
                for spin in (0, 1):
                    one[2*p+spin][2*q+spin] = self.h1[p][q]
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        value = self.h2[p][q][r][s]
                        if abs(value) <= 1e-15:
                            continue
                        for sigma in (0, 1):
                            for tau in (0, 1):
                                P, R = 2*p+sigma, 2*r+tau
                                Q, S = 2*q+sigma, 2*s+tau
                                two[P][R][Q][S] += value
                                if sigma == tau:
                                    two[P][R][S][Q] -= value
        return one, two

    def to_dict(self) -> dict[str, Any]:
        def keyed(d: dict[tuple[int, int], float]) -> dict[str, float]:
            return {f"{i},{j}": v for (i, j), v in d.items()}
        return {
            "integral_convention": "chemist_(pq|rs)",
            "n_orbitals": self.n_orbitals,
            "n_spin_orbitals": self.n_spin_orbitals,
            "n_electrons": self.n_electrons,
            "constant": self.constant,
            "spin": self.spin,
            "energy_unit": self.energy_unit,
            "h1": self.h1,
            "h2": self.h2,
            "hopping_t": keyed(self.hopping_t),
            "onsite_u": self.onsite_u,
            "intersite_v": keyed(self.intersite_v),
            "exchange_j": keyed(self.exchange_j),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialHamiltonian":
        convention = data.get("integral_convention", "chemist_(pq|rs)")
        if convention != "chemist_(pq|rs)":
            raise ValueError(f"Unsupported integral convention {convention!r}")
        def unkey(d: dict[str, float]) -> dict[tuple[int, int], float]:
            return {tuple(map(int, key.split(","))): float(value) for key, value in d.items()}
        return cls(
            n_orbitals=int(data["n_orbitals"]),
            n_electrons=float(data.get("n_electrons", 0.0)),
            constant=float(data.get("constant", 0.0)),
            spin=int(data.get("spin", 0)),
            energy_unit=data.get("energy_unit", "eV"),
            h1=data.get("h1", []),
            h2=data.get("h2", []),
            hopping_t=unkey(data.get("hopping_t", {})),
            onsite_u=data.get("onsite_u", []),
            intersite_v=unkey(data.get("intersite_v", {})),
            exchange_j=unkey(data.get("exchange_j", {})),
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        one, two = self.count_nonzero_integrals()
        return "\n".join([
            "MaterialHamiltonian Summary",
            "=" * 40,
            f"Spatial Orbitals  : {self.n_orbitals}",
            f"Spin Orbitals     : {self.n_spin_orbitals}",
            f"Active Electrons  : {self.n_electrons:.8g}",
            f"Energy Unit       : {self.energy_unit}",
            f"Integral Convention: (pq|rs)",
            f"Constant Energy   : {self.constant:.8g} {self.energy_unit}",
            f"1-Body Integrals  : {one} non-zero elements",
            f"2-Body Integrals  : {two} non-zero elements",
            f"Hermitian Valid   : {self.is_hermitian()}",
            f"Model Kind        : {self.metadata.get('model_kind', self.metadata.get('model_type', 'unspecified'))}",
        ])


def build_integral_hamiltonian(
    h1: Sequence[Sequence[float]],
    h2: Sequence[Sequence[Sequence[Sequence[float]]]],
    *,
    n_electrons: float,
    constant: float = 0.0,
    spin: int = 0,
    energy_unit: str = "Hartree",
    metadata: dict[str, Any] | None = None,
) -> MaterialHamiltonian:
    """Construct the physical finite-Hamiltonian interchange object from explicit integrals."""
    n = len(h1)
    ham = MaterialHamiltonian(
        n_orbitals=n,
        n_electrons=n_electrons,
        constant=constant,
        spin=spin,
        energy_unit=energy_unit,
        h1=[[float(v) for v in row] for row in h1],
        h2=[[[[float(v) for v in row4] for row4 in row3] for row3 in row2] for row2 in h2],
        metadata={"model_kind": "explicit_integrals", **(metadata or {})},
    )
    if not ham.is_hermitian():
        raise ValueError("explicit integral Hamiltonian violates real restricted integral symmetries")
    return ham


def build_hubbard_hamiltonian(
    n_orbitals: int,
    n_electrons: float,
    hopping_t: dict[tuple[int, int], float] | list[list[float]] | None = None,
    onsite_u: float | list[float] | None = None,
    intersite_v: dict[tuple[int, int], float] | None = None,
    constant: float = 0.0,
    energy_unit: str = "eV",
) -> MaterialHamiltonian:
    """Build a parameterized restricted Hubbard/extended-Hubbard model."""
    n = n_orbitals
    h1 = [[0.0] * n for _ in range(n)]
    h2 = [[[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    hop: dict[tuple[int, int], float] = {}
    if isinstance(hopping_t, dict):
        hop = {(int(i), int(j)): float(v) for (i, j), v in hopping_t.items()}
    elif isinstance(hopping_t, list):
        if len(hopping_t) != n or any(len(row) != n for row in hopping_t):
            raise ValueError("hopping_t matrix must be square with n_orbitals rows")
        hop = {(i, j): float(hopping_t[i][j]) for i in range(n) for j in range(n) if abs(hopping_t[i][j]) > 1e-12}
    for (i, j), value in hop.items():
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError("hopping index outside orbital range")
        h1[i][j] = value if i == j else -value
    if any(abs(h1[i][j] - h1[j][i]) > 1e-12 for i in range(n) for j in range(n)):
        raise ValueError("hopping_t must define a Hermitian/symmetric real hopping matrix")

    if onsite_u is None:
        u = [0.0] * n
    elif isinstance(onsite_u, (int, float)):
        u = [float(onsite_u)] * n
    else:
        u = [float(v) for v in onsite_u]
        if len(u) != n:
            raise ValueError("onsite_u list must contain n_orbitals values")
    for i, value in enumerate(u):
        h2[i][i][i][i] = value

    vdict = dict(intersite_v or {})
    for (i, j), value in vdict.items():
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError("intersite_v index outside orbital range")
        h2[i][i][j][j] = float(value)
        h2[j][j][i][i] = float(value)

    return MaterialHamiltonian(
        n_orbitals=n,
        n_electrons=n_electrons,
        constant=constant,
        energy_unit=energy_unit,
        h1=h1,
        h2=h2,
        hopping_t=hop,
        onsite_u=u,
        intersite_v=vdict,
        metadata={"model_kind": "parameterized_hubbard", "ab_initio": False},
    )


def _weighted_average(rows: list[list[float]], band: int, weights: list[float]) -> float:
    values = [(row[band], weights[k]) for k, row in enumerate(rows) if band < len(row)]
    if not values:
        return 0.0
    denom = sum(w for _, w in values)
    return sum(v * w for v, w in values) / denom if denom else 0.0


def build_band_model_hamiltonian(
    state: QEElectronicState | QERunResult,
    active_space: ActiveSpace,
    onsite_u_ev: float = 0.0,
    intersite_v_ev: float = 0.0,
) -> MaterialHamiltonian:
    """Construct an explicitly *heuristic* band-derived effective model.

    This function does **not** convert a QE Kohn-Sham calculation into an
    ab-initio FCIDUMP Hamiltonian.  It takes selected band indices, uses the
    k-weighted mean Kohn-Sham eigenvalue as a diagonal one-body model, and adds
    caller-specified U/V parameters.  It is intended for workflow plumbing,
    toy models, and controller tests only.

    A physical materials route should provide localized/downfolded integrals,
    e.g. QE -> Wannier90 -> cRPA/interaction construction ->
    :func:`build_integral_hamiltonian`.
    """
    el = state.electronic if isinstance(state, QERunResult) else state
    if el.lsda or el.noncolin or el.spinorbit:
        raise NotImplementedError("band-model builder supports only restricted collinear QE results")
    if not el.eigenvalues_ev:
        raise ValueError("QE eigenvalues are required for the band-derived model")
    n = active_space.n_active_orbitals
    if n <= 0:
        raise ValueError("active space is empty")
    weights = el.normalized_kpoint_weights() or [1.0 / len(el.eigenvalues_ev)] * len(el.eigenvalues_ev)
    h1 = [[0.0] * n for _ in range(n)]
    for local, band in enumerate(active_space.active_orbitals):
        h1[local][local] = _weighted_average(el.eigenvalues_ev, band, weights)
    ham = build_hubbard_hamiltonian(
        n_orbitals=n,
        n_electrons=active_space.n_active_electrons,
        hopping_t=h1,  # diagonal entries pass through as onsite energies
        onsite_u=onsite_u_ev,
        intersite_v={(i, i + 1): intersite_v_ev for i in range(n - 1)} if abs(intersite_v_ev) > 1e-12 else None,
        constant=0.0,
        energy_unit="eV",
    )
    ham.metadata.update({
        "model_kind": "qe_band_heuristic",
        "scientific_status": "experimental_heuristic",
        "ab_initio": False,
        "active_space_method": active_space.method,
        "active_bands": list(active_space.active_orbitals),
        "dft_total_energy_ev": el.total_energy_ev,
        "one_body_source": "k-weighted mean Kohn-Sham eigenvalues",
        "interaction_source": "user supplied U/V parameters",
        "warning": "Not an ab-initio QE-to-FCIDUMP conversion; no Wannier/cRPA integral construction or DFT double-counting correction is performed.",
    })
    return ham


def build_active_space_hamiltonian(*args: Any, **kwargs: Any) -> MaterialHamiltonian:
    """Deprecated compatibility alias for :func:`build_band_model_hamiltonian`."""
    warnings.warn(
        "build_active_space_hamiltonian is a band-derived heuristic, not an ab-initio QE Hamiltonian. "
        "Use build_band_model_hamiltonian for the heuristic path or build_integral_hamiltonian for physical integrals.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_band_model_hamiltonian(*args, **kwargs)
