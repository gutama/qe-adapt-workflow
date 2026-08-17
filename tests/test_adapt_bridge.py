"""Tests for QuantumSolver interface, ExactDiagonalizationSolver, and ADAPT-VQE bridge."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum.active_space import select_active_space
from qeanalyzer.quantum.adapt_bridge import (
    ADAPTVQESolver,
    ExactDiagonalizationSolver,
    QuantumRunResult,
    create_quantum_solver,
    solve_active_space,
)
from qeanalyzer.quantum.hamiltonian import (
    MaterialHamiltonian,
    build_active_space_hamiltonian,
    build_hubbard_hamiltonian,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestQuantumRunResult(unittest.TestCase):
    """Test QuantumRunResult data structure and serialization."""

    def test_serialization_roundtrip(self):
        res = QuantumRunResult(
            energy_ev=-15.456,
            electronic_energy_ev=-25.456,
            constant_energy_ev=10.0,
            correlation_energy_ev=-0.85,
            solver_type="adapt_vqe",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            converged=True,
            selected_operators=["E_{0,1}"],
            operator_gradients=[0.0005],
            operator_parameters=[0.12],
            iteration_energies=[-15.0, -15.456],
            one_rdm=[[1.0, 0.2], [0.2, 1.0]],
            natural_occupations=[1.2, 0.8],
            metadata={"pool_size": 1},
        )
        d = res.to_dict()
        rec = QuantumRunResult.from_dict(d)

        self.assertAlmostEqual(rec.energy_ev, -15.456)
        self.assertAlmostEqual(rec.electronic_energy_ev, -25.456)
        self.assertEqual(rec.solver_type, "adapt_vqe")
        self.assertEqual(rec.selected_operators, ["E_{0,1}"])
        self.assertEqual(rec.one_rdm, [[1.0, 0.2], [0.2, 1.0]])
        self.assertEqual(rec.natural_occupations, [1.2, 0.8])
        self.assertIn("Total Ground Energy: -15.456000 eV", rec.summary())


class TestExactDiagonalizationSolver(unittest.TestCase):
    """Test Exact Configuration Interaction (FCI / ED) solver."""

    def test_analytical_hubbard_dimer_ground_state(self):
        # Half-filled symmetric Hubbard dimer: t=1.0, U=4.0, N=2 electrons (1 alpha, 1 beta)
        # Analytical ground state electronic energy: E_gs = (U - sqrt(U^2 + 16*t^2)) / 2
        t = 1.0
        u = 4.0
        e_exact_analytical = (u - math.sqrt(u**2 + 16.0 * t**2)) / 2.0  # 2 - sqrt(32) ≈ -0.82842712 eV

        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): t, (1, 0): t},
            onsite_u=u,
        )

        solver = ExactDiagonalizationSolver()
        result = solver.solve(ham)

        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.electronic_energy_ev, e_exact_analytical, places=6)
        self.assertAlmostEqual(result.energy_ev, e_exact_analytical, places=6)

        # Verify 1-RDM: trace must equal N_electrons = 2.0
        tr = sum(result.one_rdm[i][i] for i in range(2))
        self.assertAlmostEqual(tr, 2.0, places=6)

        # Symmetry: site occupations are equal (1.0 each)
        self.assertAlmostEqual(result.one_rdm[0][0], 1.0, places=6)
        self.assertAlmostEqual(result.one_rdm[1][1], 1.0, places=6)

        # Off-diagonal hopping density matrix element
        self.assertGreater(abs(result.one_rdm[0][1]), 0.0)

        # Natural occupations sum to 2.0
        self.assertAlmostEqual(sum(result.natural_occupations), 2.0, places=6)

    def test_dft_silicon_active_space_solution(self):
        xml = read_qe_xml(FIXTURES / "si_scf.xml")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        res = build_run_result(pw_out=pw_out, qe_xml=xml, run_id="si-scf")

        asp = select_active_space(res, method="band_index", band_start=1, band_end=2)
        ham = build_active_space_hamiltonian(res, active_space=asp, onsite_u_ev=2.0)

        solver = ExactDiagonalizationSolver()
        q_res = solver.solve(ham, active_space=asp)

        self.assertTrue(q_res.converged)
        self.assertEqual(q_res.n_orbitals, 2)
        self.assertAlmostEqual(q_res.constant_energy_ev, ham.constant)
        self.assertAlmostEqual(sum(q_res.natural_occupations), ham.n_electrons, places=4)


class TestADAPTVQESolver(unittest.TestCase):
    """Test ADAPT-VQE algorithmic bridge and iteration growth."""

    def test_adapt_vqe_hubbard_dimer(self):
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): 1.0, (1, 0): 1.0},
            onsite_u=3.0,
        )

        solver = ADAPTVQESolver(gradient_threshold=1e-3, max_adapt_iterations=10)
        res = solver.solve(ham)

        self.assertEqual(res.solver_type, "adapt_vqe")
        self.assertTrue(res.converged)
        self.assertGreater(len(res.selected_operators), 0)
        self.assertEqual(len(res.selected_operators), len(res.operator_parameters))
        self.assertEqual(len(res.iteration_energies), len(res.selected_operators) + 1)

        # Energy decreases monotonically with ADAPT operator growth
        for i in range(len(res.iteration_energies) - 1):
            self.assertLessEqual(res.iteration_energies[i + 1], res.iteration_energies[i] + 1e-6)

        # Correlation energy is negative
        self.assertIsNotNone(res.correlation_energy_ev)
        self.assertLess(res.correlation_energy_ev, 0.0)


class TestSolverFactoryAndConvenience(unittest.TestCase):
    """Test solver instantiation and high-level execution."""

    def test_create_solver_factory(self):
        s_exact = create_quantum_solver("exact")
        self.assertIsInstance(s_exact, ExactDiagonalizationSolver)

        s_adapt = create_quantum_solver("adapt_vqe")
        self.assertIsInstance(s_adapt, ADAPTVQESolver)

        with self.assertRaises(ValueError):
            create_quantum_solver("invalid_solver")

    def test_solve_active_space_convenience(self):
        ham = build_hubbard_hamiltonian(n_orbitals=2, n_electrons=2.0, hopping_t={(0, 1): 1.0, (1, 0): 1.0}, onsite_u=2.0)
        res = solve_active_space(ham, solver_type="exact")
        self.assertTrue(res.converged)
        self.assertEqual(res.n_orbitals, 2)


if __name__ == "__main__":
    unittest.main()
