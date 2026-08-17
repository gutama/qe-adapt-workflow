"""Quantum-solver interfaces and a small exact reference solver.

Scientific ADAPT-VQE is intentionally *not* implemented here.  The public
``ADAPTVQESolver`` constructor delegates to the sibling ``clifford_qc`` project.
``SimulatedADAPTVQESolver`` is retained only as an explicitly named workflow
mock.
"""

from __future__ import annotations

import itertools
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qeanalyzer.quantum.active_space import ActiveSpace
from qeanalyzer.quantum.hamiltonian import MaterialHamiltonian
from qeanalyzer.quantum.units import HARTREE_TO_EV, energy_to_hartree, require_integer_electron_sector


@dataclass
class QuantumRunResult:
    energy_ev: float
    electronic_energy_ev: float
    solver_type: str
    n_orbitals: int
    n_electrons: float
    n_spin_orbitals: int
    converged: bool = True
    constant_energy_ev: float = 0.0
    correlation_energy_ev: float | None = None
    selected_operators: list[str] = field(default_factory=list)
    operator_gradients: list[float] = field(default_factory=list)
    operator_parameters: list[float] = field(default_factory=list)
    iteration_energies: list[float] = field(default_factory=list)
    one_rdm: list[list[float]] = field(default_factory=list)
    natural_occupations: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy_ev": self.energy_ev,
            "electronic_energy_ev": self.electronic_energy_ev,
            "constant_energy_ev": self.constant_energy_ev,
            "correlation_energy_ev": self.correlation_energy_ev,
            "solver_type": self.solver_type,
            "n_orbitals": self.n_orbitals,
            "n_electrons": self.n_electrons,
            "n_spin_orbitals": self.n_spin_orbitals,
            "converged": self.converged,
            "selected_operators": list(self.selected_operators),
            "operator_gradients": list(self.operator_gradients),
            "operator_parameters": list(self.operator_parameters),
            "iteration_energies": list(self.iteration_energies),
            "one_rdm": self.one_rdm,
            "natural_occupations": list(self.natural_occupations),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantumRunResult":
        return cls(
            energy_ev=data["energy_ev"],
            electronic_energy_ev=data.get("electronic_energy_ev", data["energy_ev"]),
            constant_energy_ev=data.get("constant_energy_ev", 0.0),
            correlation_energy_ev=data.get("correlation_energy_ev"),
            solver_type=data.get("solver_type", "exact_diagonalization"),
            n_orbitals=data["n_orbitals"],
            n_electrons=data.get("n_electrons", 0.0),
            n_spin_orbitals=data.get("n_spin_orbitals", 2 * data["n_orbitals"]),
            converged=data.get("converged", True),
            selected_operators=data.get("selected_operators", []),
            operator_gradients=data.get("operator_gradients", []),
            operator_parameters=data.get("operator_parameters", []),
            iteration_energies=data.get("iteration_energies", []),
            one_rdm=data.get("one_rdm", []),
            natural_occupations=data.get("natural_occupations", []),
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        return "\n".join([
            f"Quantum Calculation [{self.solver_type}]",
            "=" * 40,
            f"Total Ground Energy: {self.energy_ev:.8f} eV",
            f"Electronic Energy  : {self.electronic_energy_ev:.8f} eV",
            f"Active (orb / elec): {self.n_orbitals} / {self.n_electrons:.8g}",
            f"Converged          : {self.converged}",
        ])


class QuantumSolver(ABC):
    @abstractmethod
    def solve(self, hamiltonian: MaterialHamiltonian, active_space: ActiveSpace | None = None,
              initial_state: Any = None, **kwargs: Any) -> QuantumRunResult:
        ...


def _fermion_apply(state: int, orbital: int, create: bool) -> tuple[int, int] | None:
    occupied = (state >> orbital) & 1
    if create:
        if occupied:
            return None
    elif not occupied:
        return None
    phase = -1 if ((state & ((1 << orbital) - 1)).bit_count() & 1) else 1
    if create:
        state |= 1 << orbital
    else:
        state &= ~(1 << orbital)
    return state, phase


def _apply_string(state: int, operations_right_to_left: list[tuple[int, bool]]) -> tuple[int, int] | None:
    phase = 1
    current = state
    for orbital, create in operations_right_to_left:
        result = _fermion_apply(current, orbital, create)
        if result is None:
            return None
        current, sign = result
        phase *= sign
    return current, phase


def _sector_determinants(n_orbitals: int, n_alpha: int, n_beta: int) -> list[int]:
    states: list[int] = []
    for alpha in itertools.combinations(range(n_orbitals), n_alpha):
        for beta in itertools.combinations(range(n_orbitals), n_beta):
            bits = 0
            for p in alpha:
                bits |= 1 << (2 * p)
            for p in beta:
                bits |= 1 << (2 * p + 1)
            states.append(bits)
    return states


class ExactDiagonalizationSolver(QuantumSolver):
    """Small-space exact FCI reference via explicit second-quantized operators.

    This implementation is intentionally straightforward rather than optimized;
    it is an independent correctness/reference path for small active spaces.
    """

    def solve(self, hamiltonian: MaterialHamiltonian, active_space: ActiveSpace | None = None,
              initial_state: Any = None, **kwargs: Any) -> QuantumRunResult:
        if initial_state is not None:
            raise NotImplementedError("custom initial states are not supported by the exact reference solver")
        if not hamiltonian.is_hermitian():
            raise ValueError("Hamiltonian is not Hermitian under the restricted (pq|rs) convention")
        n = hamiltonian.n_orbitals
        nelec = require_integer_electron_sector(hamiltonian.n_electrons)
        ms2 = int(hamiltonian.spin)
        if abs(ms2) > nelec or (nelec + ms2) % 2:
            raise ValueError(f"Incompatible NELEC={nelec} and MS2={ms2}")
        n_alpha = (nelec + ms2) // 2
        n_beta = nelec - n_alpha
        if n_alpha > n or n_beta > n:
            raise ValueError("Requested electron/spin sector does not fit the active space")

        determinants = _sector_determinants(n, n_alpha, n_beta)
        index = {state: i for i, state in enumerate(determinants)}
        dim = len(determinants)
        matrix = np.zeros((dim, dim), dtype=float)

        to_ev = lambda value: energy_to_hartree(value, hamiltonian.energy_unit) * HARTREE_TO_EV
        one_terms = [
            (p, q, to_ev(hamiltonian.h1[p][q]))
            for p in range(n) for q in range(n)
            if abs(hamiltonian.h1[p][q]) > 1e-14
        ]
        two_terms = [
            (p, q, r, s, 0.5 * to_ev(hamiltonian.h2[p][q][r][s]))
            for p in range(n) for q in range(n) for r in range(n) for s in range(n)
            if abs(hamiltonian.h2[p][q][r][s]) > 1e-14
        ]

        for col, ket in enumerate(determinants):
            for p, q, coefficient in one_terms:
                for spin in (0, 1):
                    P, Q = 2 * p + spin, 2 * q + spin
                    applied = _apply_string(ket, [(Q, False), (P, True)])
                    if applied is not None:
                        out, sign = applied
                        row = index.get(out)
                        if row is not None:
                            matrix[row, col] += coefficient * sign
            for p, q, r, s, coefficient in two_terms:
                for sigma in (0, 1):
                    for tau in (0, 1):
                        P, Q = 2 * p + sigma, 2 * q + sigma
                        R, S = 2 * r + tau, 2 * s + tau
                        applied = _apply_string(
                            ket,
                            [(Q, False), (S, False), (R, True), (P, True)],
                        )
                        if applied is not None:
                            out, sign = applied
                            row = index.get(out)
                            if row is not None:
                                matrix[row, col] += coefficient * sign

        if not np.allclose(matrix, matrix.T, atol=1e-9, rtol=1e-9):
            raise ValueError("Second-quantized assembly produced a non-Hermitian CI matrix")
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        electronic = float(eigenvalues[0])
        coeffs = eigenvectors[:, 0]

        one_rdm = np.zeros((n, n), dtype=float)
        for p in range(n):
            for q in range(n):
                value = 0.0
                for col, ket in enumerate(determinants):
                    c_ket = coeffs[col]
                    for spin in (0, 1):
                        P, Q = 2 * p + spin, 2 * q + spin
                        applied = _apply_string(ket, [(Q, False), (P, True)])
                        if applied is None:
                            continue
                        out, sign = applied
                        row = index.get(out)
                        if row is not None:
                            value += float(coeffs[row] * c_ket * sign)
                one_rdm[p, q] = value
        one_rdm = 0.5 * (one_rdm + one_rdm.T)
        natural = sorted((float(x) for x in np.linalg.eigvalsh(one_rdm)), reverse=True)
        constant_ev = to_ev(hamiltonian.constant)
        total = electronic + constant_ev
        return QuantumRunResult(
            energy_ev=round(total, 10),
            electronic_energy_ev=round(electronic, 10),
            constant_energy_ev=round(constant_ev, 10),
            solver_type="exact_diagonalization",
            n_orbitals=n,
            n_electrons=float(nelec),
            n_spin_orbitals=2 * n,
            converged=True,
            one_rdm=one_rdm.tolist(),
            natural_occupations=[round(x, 10) for x in natural],
            metadata={"fci_dimension": dim, "integral_convention": "chemist_(pq|rs)"},
        )


class SimulatedADAPTVQESolver(QuantumSolver):
    """Non-scientific monotone trajectory for workflow plumbing tests only."""

    def __init__(self, gradient_threshold: float = 1e-3, max_adapt_iterations: int = 20) -> None:
        warnings.warn(
            "SimulatedADAPTVQESolver is a workflow mock, not an ADAPT-VQE implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        self.gradient_threshold = float(gradient_threshold)
        self.max_adapt_iterations = int(max_adapt_iterations)

    def solve(self, hamiltonian: MaterialHamiltonian, active_space: ActiveSpace | None = None,
              initial_state: Any = None, **kwargs: Any) -> QuantumRunResult:
        exact = ExactDiagonalizationSolver().solve(hamiltonian, active_space=active_space)
        target = exact.electronic_energy_ev
        current = target + max(0.25 * abs(target), 0.5)
        constant_ev = exact.constant_energy_ev
        energies = [current + constant_ev]
        labels: list[str] = []
        gradients: list[float] = []
        parameters: list[float] = []
        for step in range(self.max_adapt_iterations):
            error = max(0.0, current - target)
            gradient = error * (0.8 ** (step + 1))
            gradients.append(gradient)
            if gradient < self.gradient_threshold:
                break
            labels.append(f"MOCK_E_{step}")
            parameters.append(math.atan(gradient))
            current = target + 0.45 * error
            energies.append(current + constant_ev)
        return QuantumRunResult(
            energy_ev=round(current + constant_ev, 10),
            electronic_energy_ev=round(current, 10),
            constant_energy_ev=round(constant_ev, 10),
            solver_type="simulated_adapt_workflow_mock",
            n_orbitals=hamiltonian.n_orbitals,
            n_electrons=hamiltonian.n_electrons,
            n_spin_orbitals=hamiltonian.n_spin_orbitals,
            converged=bool(gradients and gradients[-1] < self.gradient_threshold),
            selected_operators=labels,
            operator_gradients=[],  # deliberately absent: not a physical commutator residual
            operator_parameters=parameters,
            iteration_energies=energies,
            one_rdm=exact.one_rdm,
            natural_occupations=exact.natural_occupations,
            metadata={
                "scientific_status": "workflow_mock",
                "residual_gradient": None,
                "uses_exact_solution_to_generate_mock": True,
            },
        )


class ADAPTVQESolver:
    """Compatibility constructor for the real ``clifford_qc`` ADAPT backend."""

    def __new__(cls, *args: Any, **kwargs: Any):
        from qeanalyzer.quantum.clifford_bridge import CliffordQCADAPTSolver
        return CliffordQCADAPTSolver(*args, **kwargs)


def create_quantum_solver(solver_type: str = "exact", **kwargs: Any) -> QuantumSolver:
    key = solver_type.lower().replace("-", "_")
    if key in {"exact", "fci", "exact_diagonalization", "ed"}:
        return ExactDiagonalizationSolver()
    if key in {"adapt", "adapt_vqe", "clifford_adapt", "clifford_qc_adapt"}:
        return ADAPTVQESolver(**kwargs)
    if key in {"simulated_adapt", "mock_adapt", "workflow_mock"}:
        return SimulatedADAPTVQESolver(**kwargs)
    raise ValueError(f"Unknown quantum solver type {solver_type!r}")


def solve_active_space(hamiltonian: MaterialHamiltonian, active_space: ActiveSpace | None = None,
                       solver_type: str = "exact", **kwargs: Any) -> QuantumRunResult:
    return create_quantum_solver(solver_type, **kwargs).solve(hamiltonian, active_space=active_space)
