"""Stable quantum-solver facade for QE-ADAPT workflow integration.

This module is the single place that maps a solver name onto a solver class.
The actual ADAPT-VQE implementation is owned by the sibling ``clifford_qc``
project, so this module deliberately contains no second ADAPT implementation --
and no second copy of the factory either.
"""

from typing import Any

from qeanalyzer.quantum.adapt_bridge import (
    ExactDiagonalizationSolver,
    QuantumRunResult,
    QuantumSolver,
    SimulatedADAPTVQESolver,
)
from qeanalyzer.quantum.clifford_bridge import CliffordQCADAPTSolver
from qeanalyzer.quantum.hamiltonian import MaterialHamiltonian

# Backwards-compatible public name: real ADAPT, delegated to clifford_qc. It is
# the class itself, so isinstance(solver, ADAPTVQESolver) behaves.
ADAPTVQESolver = CliffordQCADAPTSolver

EXACT_ALIASES = frozenset({"exact", "fci", "exact_diagonalization", "ed"})
ADAPT_ALIASES = frozenset({"adapt", "adapt_vqe", "vqe", "clifford_adapt", "clifford_qc_adapt"})
MOCK_ALIASES = frozenset({"simulated_adapt", "mock_adapt", "workflow_mock"})


def create_quantum_solver(solver_type: str = "exact", **solver_options: Any) -> QuantumSolver:
    """Create a solver without duplicating ADAPT-VQE inside this repository.

    ``adapt``/``adapt_vqe`` selects the real :mod:`clifford_qc` backend.
    ``simulated_adapt`` is an explicit workflow-only mock and must never be used
    as scientific evidence.  Names are normalized, so ``adapt-vqe`` and
    ``adapt_vqe`` are the same solver.

    ``solver_options`` are the chosen solver's constructor arguments.  The exact
    solver has none, so passing any is an error rather than a silent no-op.
    """
    key = solver_type.lower().replace("-", "_")
    if key in EXACT_ALIASES:
        if solver_options:
            raise TypeError(
                f"solver_type={solver_type!r} takes no options, but got "
                f"{sorted(solver_options)}. Exact diagonalization has nothing to tune; "
                "pass solver_type='adapt_vqe' if these are ADAPT settings."
            )
        return ExactDiagonalizationSolver()
    if key in ADAPT_ALIASES:
        return CliffordQCADAPTSolver(**solver_options)
    if key in MOCK_ALIASES:
        return SimulatedADAPTVQESolver(**solver_options)
    raise ValueError(
        f"Unknown quantum solver type {solver_type!r}. "
        "Choose 'exact', 'adapt_vqe', or 'simulated_adapt'."
    )


def solve_active_space(
    hamiltonian: MaterialHamiltonian,
    active_space: Any = None,
    solver_type: str = "exact",
    **solver_options: Any,
) -> QuantumRunResult:
    """Build the named solver and run it once.

    ``solver_options`` configure the solver itself and are forwarded only to its
    constructor.  Forwarding them to ``solve`` as well meant an option the
    constructor ignored was swallowed by ``solve(**kwargs)``, so a caller's
    setting could take effect in neither place without raising.
    """
    solver = create_quantum_solver(solver_type, **solver_options)
    return solver.solve(hamiltonian, active_space=active_space)


__all__ = [
    "ADAPTVQESolver",
    "ADAPT_ALIASES",
    "CliffordQCADAPTSolver",
    "EXACT_ALIASES",
    "ExactDiagonalizationSolver",
    "MOCK_ALIASES",
    "QuantumRunResult",
    "QuantumSolver",
    "SimulatedADAPTVQESolver",
    "create_quantum_solver",
    "solve_active_space",
]
