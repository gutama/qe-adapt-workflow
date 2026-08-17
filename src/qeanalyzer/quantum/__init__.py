"""Quantum-model interchange and solver interfaces for QE-ADAPT workflows."""

from qeanalyzer.quantum.active_space import (
    ActiveSpace,
    ActiveSpaceSelector,
    BandIndexSelector,
    EnergyWindowSelector,
    ExplicitOrbitalSelector,
    OccupationSelector,
    create_active_space_selector,
    select_active_space,
)
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult, QuantumSolver
from qeanalyzer.quantum.clifford_bridge import CliffordQCADAPTSolver, clifford_qc_available
from qeanalyzer.quantum.fcidump import parse_fcidump, read_fcidump, write_fcidump
from qeanalyzer.quantum.feedback import (
    ActiveSpaceFeedbackPolicy,
    HubbardUFeedbackPolicy,
    OccupationFeedbackPolicy,
    QuantumFeedbackPolicy,
    apply_quantum_feedback,
)
from qeanalyzer.quantum.hamiltonian import (
    MaterialHamiltonian,
    build_active_space_hamiltonian,
    build_band_model_hamiltonian,
    build_hubbard_hamiltonian,
    build_integral_hamiltonian,
)
from qeanalyzer.quantum.solver_api import (
    ADAPTVQESolver,
    ExactDiagonalizationSolver,
    SimulatedADAPTVQESolver,
    create_quantum_solver,
    solve_active_space,
)

__all__ = [
    "ADAPTVQESolver",
    "ActiveSpace",
    "ActiveSpaceFeedbackPolicy",
    "ActiveSpaceSelector",
    "BandIndexSelector",
    "CliffordQCADAPTSolver",
    "EnergyWindowSelector",
    "ExactDiagonalizationSolver",
    "ExplicitOrbitalSelector",
    "HubbardUFeedbackPolicy",
    "MaterialHamiltonian",
    "OccupationFeedbackPolicy",
    "OccupationSelector",
    "QuantumFeedbackPolicy",
    "QuantumRunResult",
    "QuantumSolver",
    "SimulatedADAPTVQESolver",
    "apply_quantum_feedback",
    "build_active_space_hamiltonian",
    "build_band_model_hamiltonian",
    "build_hubbard_hamiltonian",
    "build_integral_hamiltonian",
    "clifford_qc_available",
    "create_active_space_selector",
    "create_quantum_solver",
    "parse_fcidump",
    "read_fcidump",
    "select_active_space",
    "solve_active_space",
    "write_fcidump",
]
