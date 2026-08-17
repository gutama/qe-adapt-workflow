"""Stable quantum-solver facade for QE-ADAPT workflow integration.

The actual ADAPT-VQE implementation is owned by the sibling ``clifford_qc``
project.  This module deliberately contains no second ADAPT implementation.
"""

from qeanalyzer.quantum.adapt_bridge import (
    ExactDiagonalizationSolver,
    QuantumRunResult,
    QuantumSolver,
    SimulatedADAPTVQESolver,
    solve_active_space as _solve_active_space,
)
from qeanalyzer.quantum.clifford_bridge import CliffordQCADAPTSolver

# Backwards-compatible public name: real ADAPT, delegated to clifford_qc.
ADAPTVQESolver = CliffordQCADAPTSolver


def create_quantum_solver(solver_type: str = "exact", **kwargs):
    """Create a solver without duplicating ADAPT-VQE inside this repository.

    ``adapt``/``adapt_vqe`` selects the real :mod:`clifford_qc` backend.
    ``simulated_adapt`` is an explicit workflow-only mock and must never be used
    as scientific evidence.
    """
    st = solver_type.lower()
    if st in ("exact", "fci", "exact_diagonalization", "ed"):
        return ExactDiagonalizationSolver()
    if st in ("adapt", "adapt_vqe", "vqe", "clifford_adapt"):
        return CliffordQCADAPTSolver(**kwargs)
    if st in ("simulated_adapt", "mock_adapt", "workflow_mock"):
        return SimulatedADAPTVQESolver(**kwargs)
    raise ValueError(
        f"Unknown quantum solver type {solver_type!r}. "
        "Choose 'exact', 'adapt_vqe', or 'simulated_adapt'."
    )


def solve_active_space(hamiltonian, active_space=None, solver_type: str = "exact", **kwargs):
    solver = create_quantum_solver(solver_type=solver_type, **kwargs)
    return solver.solve(hamiltonian=hamiltonian, active_space=active_space, **kwargs)


__all__ = [
    "ADAPTVQESolver",
    "CliffordQCADAPTSolver",
    "ExactDiagonalizationSolver",
    "QuantumRunResult",
    "QuantumSolver",
    "SimulatedADAPTVQESolver",
    "create_quantum_solver",
    "solve_active_space",
]
