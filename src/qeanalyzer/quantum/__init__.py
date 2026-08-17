"""Quantum solver bridge and active space interfaces for DFT-ADAPT-VQE coupling."""

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
from qeanalyzer.quantum.adapt_bridge import (
    ADAPTVQESolver,
    ExactDiagonalizationSolver,
    QuantumRunResult,
    QuantumSolver,
    create_quantum_solver,
    solve_active_space,
)
from qeanalyzer.quantum.fcidump import (
    parse_fcidump,
    read_fcidump,
    write_fcidump,
)
from qeanalyzer.quantum.hamiltonian import (
    MaterialHamiltonian,
    build_active_space_hamiltonian,
    build_hubbard_hamiltonian,
)

__all__ = [
    "ADAPTVQESolver",
    "ActiveSpace",
    "ActiveSpaceSelector",
    "BandIndexSelector",
    "EnergyWindowSelector",
    "ExactDiagonalizationSolver",
    "ExplicitOrbitalSelector",
    "MaterialHamiltonian",
    "OccupationSelector",
    "QuantumRunResult",
    "QuantumSolver",
    "build_active_space_hamiltonian",
    "build_hubbard_hamiltonian",
    "create_active_space_selector",
    "create_quantum_solver",
    "parse_fcidump",
    "read_fcidump",
    "select_active_space",
    "solve_active_space",
    "write_fcidump",
]
