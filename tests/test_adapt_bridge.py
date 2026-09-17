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
    solve_active_space,
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


class TestFCISameSpinDoubleExcitations(unittest.TestCase):
    """Regression tests for alpha-alpha / beta-beta double excitations.

    The other FCI tests in this file all use two electrons (one alpha, one
    beta), where no same-spin double excitation exists, and Hubbard-style
    density-density integrals, where every same-spin double element vanishes.
    Neither constrains this part of the CI matrix.
    """

    @staticmethod
    def _model_hamiltonian(n_electrons: float, spin: int = 0) -> MaterialHamiltonian:
        """A 4-orbital model with general (non density-density) two-body integrals.

        ``h2`` is built in factorized form so it carries the full 8-fold
        permutational symmetry of a real ``(ij|kl)`` integral tensor.
        """
        norb = 4
        h1 = [[0.0] * norb for _ in range(norb)]
        for i in range(norb):
            h1[i][i] = 0.25 * i
            for j in range(norb):
                if abs(i - j) == 1:
                    h1[i][j] = -1.0
        b = [
            [[1.0 / (1.0 + a + i + j) for j in range(norb)] for i in range(norb)]
            for a in range(2)
        ]
        h2 = [
            [
                [
                    [sum(b[a][i][j] * b[a][k][l] for a in range(2)) for l in range(norb)]
                    for k in range(norb)
                ]
                for j in range(norb)
            ]
            for i in range(norb)
        ]
        return MaterialHamiltonian(
            n_orbitals=norb,
            n_electrons=n_electrons,
            spin=spin,
            h1=h1,
            h2=h2,
            constant=0.0,
            energy_unit="eV",
        )

    # Reference energies come from an independent brute-force FCI that applies
    # each second-quantized operator explicitly rather than using Slater-Condon
    # rules; it was validated against the non-interacting limit (sum of the
    # lowest h1 eigenvalues) and the analytic Hubbard-dimer ground state.

    def test_two_electron_ground_state(self):
        """One alpha + one beta: no same-spin doubles exist, so this always held."""
        res = ExactDiagonalizationSolver().solve(self._model_hamiltonian(2.0))
        self.assertAlmostEqual(res.energy_ev, -1.4337384453, places=9)

    def test_four_electron_ground_state(self):
        """Two alpha + two beta: requires the same-spin double block to be present."""
        res = ExactDiagonalizationSolver().solve(self._model_hamiltonian(4.0))
        self.assertAlmostEqual(res.energy_ev, -0.9315163058, places=9)

    def test_energy_invariant_under_orbital_rotation(self):
        """The FCI energy is basis independent; a truncated CI expansion is not.

        Rotating the orbital basis and transforming h1/h2 accordingly must leave
        the ground-state energy unchanged. Dropping any class of excitation
        breaks this invariance, so it holds without reference to a stored number.
        """
        import numpy as np

        ham = self._model_hamiltonian(4.0)
        norb = ham.n_orbitals

        u = np.eye(norb)
        for i, j, theta in ((0, 1, 0.3), (1, 2, -0.45), (2, 3, 0.7), (0, 3, 0.2)):
            g = np.eye(norb)
            c, s = math.cos(theta), math.sin(theta)
            g[i, i], g[j, j], g[i, j], g[j, i] = c, c, -s, s
            u = u @ g

        h1_rot = u.T @ np.array(ham.h1) @ u
        h2_rot = np.einsum(
            "ijkl,ip,jq,kr,ls->pqrs", np.array(ham.h2), u, u, u, u, optimize=True
        )
        rotated = MaterialHamiltonian(
            n_orbitals=norb,
            n_electrons=ham.n_electrons,
            spin=ham.spin,
            h1=h1_rot.tolist(),
            h2=h2_rot.tolist(),
            constant=0.0,
            energy_unit="eV",
        )

        solver = ExactDiagonalizationSolver()
        self.assertAlmostEqual(
            solver.solve(ham).energy_ev,
            solver.solve(rotated).energy_ev,
            places=9,
        )

    def test_high_spin_state_ground_state(self):
        """Three alpha + one beta exercises the alpha-alpha block on its own."""
        res = ExactDiagonalizationSolver().solve(self._model_hamiltonian(4.0, spin=2))
        ref = ExactDiagonalizationSolver().solve(self._model_hamiltonian(4.0))
        # The high-spin sector must lie above the unrestricted ground state.
        self.assertGreater(res.energy_ev, ref.energy_ev)


class TestSolverFacade(unittest.TestCase):
    """One factory, one alias set, and options that land somewhere or raise.

    The factory used to exist twice -- here and in ``adapt_bridge`` -- with
    different aliases and a different ADAPTVQESolver, so the answer depended on
    which module the caller imported from.
    """

    @staticmethod
    def _dimer():
        return build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): 1.0, (1, 0): 1.0},
            onsite_u=2.0,
        )

    def test_adapt_aliases_all_resolve_to_the_clifford_backend(self):
        for alias in ("adapt", "adapt-vqe", "adapt_vqe", "clifford_adapt",
                      "clifford_qc_adapt", "ADAPT-VQE"):
            with self.subTest(alias=alias):
                self.assertIsInstance(create_quantum_solver(alias), CliffordQCADAPTSolver)

    def test_bare_vqe_is_no_longer_silently_adapt(self):
        """It meant ADAPT while ADAPT was the only method; now it names nothing.

        A-CASE is not a VQE and clifford_qc has a fixed-ansatz VQE of its own,
        so guessing here would label a run as the method it was not.
        """
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            create_quantum_solver("vqe")

    def test_adapt_bridge_no_longer_ships_a_second_factory(self):
        from qeanalyzer.quantum import adapt_bridge, solver_api

        for name in ("create_quantum_solver", "ADAPTVQESolver", "solve_active_space"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(adapt_bridge, name))
                self.assertTrue(hasattr(solver_api, name))

    def test_solve_active_space_matches_the_solver_it_names(self):
        ham = self._dimer()
        self.assertAlmostEqual(
            solve_active_space(ham).energy_ev,
            ExactDiagonalizationSolver().solve(ham).energy_ev,
            places=10,
        )

    def test_solver_options_reach_the_constructor(self):
        solver = create_quantum_solver("adapt_vqe", gradient_threshold=1e-9, maxiter=7)
        self.assertEqual(solver.gradient_threshold, 1e-9)
        self.assertEqual(solver.maxiter, 7)

    def test_options_the_exact_solver_cannot_use_are_rejected(self):
        """Silently dropping them let a caller's setting take effect nowhere."""
        with self.assertRaisesRegex(TypeError, "takes no options"):
            create_quantum_solver("exact", gradient_threshold=1e-6)
        with self.assertRaisesRegex(TypeError, "takes no options"):
            solve_active_space(self._dimer(), gradient_threshold=1e-6)

    def test_unknown_solver_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown quantum solver type"):
            create_quantum_solver("not_a_solver")


if __name__ == "__main__":
    unittest.main()
