"""Quantum-solver boundary tests.

Real ADAPT-VQE belongs to clifford_qc.  The local exact solver is an independent
small-space reference, while the simulated ADAPT class is explicitly a plumbing
mock.
"""

from __future__ import annotations

import math
import unittest
import warnings

from qeanalyzer.quantum import (
    ADAPTVQESolver,
    CliffordQCADAPTSolver,
    ExactDiagonalizationSolver,
    QuantumRunResult,
    SimulatedADAPTVQESolver,
    build_hubbard_hamiltonian,
    clifford_qc_available,
    create_quantum_solver,
)
from qeanalyzer.quantum.hamiltonian import MaterialHamiltonian


class TestQuantumRunResult(unittest.TestCase):
    def test_roundtrip(self):
        result = QuantumRunResult(
            energy_ev=-1.0,
            electronic_energy_ev=-2.0,
            constant_energy_ev=1.0,
            solver_type="exact_diagonalization",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            one_rdm=[[1.0, 0.1], [0.1, 1.0]],
        )
        self.assertEqual(QuantumRunResult.from_dict(result.to_dict()).one_rdm, result.one_rdm)


class TestExactSolver(unittest.TestCase):
    def test_hubbard_dimer_analytic(self):
        t, u = 1.0, 4.0
        expected = (u - math.sqrt(u * u + 16.0 * t * t)) / 2.0
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): t, (1, 0): t},
            onsite_u=u,
        )
        result = ExactDiagonalizationSolver().solve(ham)
        self.assertAlmostEqual(result.energy_ev, expected, places=7)
        self.assertAlmostEqual(sum(result.natural_occupations), 2.0, places=7)
        self.assertAlmostEqual(sum(result.one_rdm[i][i] for i in range(2)), 2.0, places=7)

    def test_hartree_input_is_reported_in_ev(self):
        ham = MaterialHamiltonian(
            n_orbitals=1,
            n_electrons=1.0,
            spin=1,
            energy_unit="Hartree",
            h1=[[1.0]],
        )
        result = ExactDiagonalizationSolver().solve(ham)
        self.assertAlmostEqual(result.energy_ev, 27.211386245988, places=8)

    def test_fractional_sector_rejected(self):
        ham = build_hubbard_hamiltonian(n_orbitals=2, n_electrons=1.4)
        with self.assertRaisesRegex(ValueError, "fractional"):
            ExactDiagonalizationSolver().solve(ham)


class TestADAPTOwnership(unittest.TestCase):
    def test_public_adapt_constructor_delegates_to_clifford_qc(self):
        solver = ADAPTVQESolver()
        self.assertIsInstance(solver, CliffordQCADAPTSolver)
        self.assertIsInstance(create_quantum_solver("adapt_vqe"), CliffordQCADAPTSolver)

    def test_simulated_solver_is_explicit_mock(self):
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): 1.0, (1, 0): 1.0},
            onsite_u=3.0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = SimulatedADAPTVQESolver(max_adapt_iterations=10)
        self.assertTrue(any("workflow mock" in str(w.message) for w in caught))
        result = solver.solve(ham)
        self.assertEqual(result.solver_type, "simulated_adapt_workflow_mock")
        self.assertEqual(result.metadata["scientific_status"], "workflow_mock")
        self.assertIsNone(result.metadata["residual_gradient"])
        self.assertEqual(result.operator_gradients, [])

    @unittest.skipUnless(clifford_qc_available(), "clifford_qc optional sibling not installed")
    def test_real_clifford_adapt_hubbard_dimer(self):
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): 1.0, (1, 0): 1.0},
            onsite_u=2.0,
        )
        result = CliffordQCADAPTSolver(
            gradient_threshold=1e-6,
            max_adapt_iterations=8,
            compute_exact_reference=True,
        ).solve(ham)
        self.assertEqual(result.solver_type, "clifford_qc_adapt_vqe")
        self.assertEqual(result.metadata["backend_project"], "gutama/clifford_qc")
        self.assertIsNotNone(result.metadata["residual_gradient"])
        self.assertAlmostEqual(sum(result.natural_occupations), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
