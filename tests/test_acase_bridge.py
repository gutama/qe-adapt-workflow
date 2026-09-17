"""A-CASE boundary tests.

A-CASE is a different algorithm from ADAPT-VQE, not a second name for it: it
grows a linear subspace around one reference and solves the projected
generalized eigenproblem.  These tests pin the properties that make its results
usable by the rest of the workflow -- variational upper bounds, a residual that
means something, a 1-RDM in the same convention as every other solver -- and the
places where the warm-started composition is honest about not converging.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest

from qeanalyzer.quantum import (
    CliffordQCACASESolver,
    CliffordQCADAPTACASESolver,
    ExactDiagonalizationSolver,
    build_hubbard_hamiltonian,
    clifford_qc_subspace_available,
    create_quantum_solver,
)

HAS_SUBSPACE = clifford_qc_subspace_available()


def hubbard_chain(n_orbitals: int, n_electrons: float, u: float = 3.0, t: float = 1.0):
    hopping = {}
    for i in range(n_orbitals - 1):
        hopping[(i, i + 1)] = t
        hopping[(i + 1, i)] = t
    return build_hubbard_hamiltonian(
        n_orbitals=n_orbitals, n_electrons=n_electrons, hopping_t=hopping, onsite_u=u
    )


@unittest.skipUnless(HAS_SUBSPACE, "clifford_qc.subspace optional sibling not installed")
class TestACASEAgainstExactFCI(unittest.TestCase):
    def test_complete_determinant_basis_reproduces_exact_fci(self):
        """Singles + doubles span the whole (N=2, Sz=0) sector of a dimer."""
        ham = hubbard_chain(2, 2.0, u=2.0)
        exact = ExactDiagonalizationSolver().solve(ham)
        result = CliffordQCACASESolver(max_basis_size=8).solve(ham)

        self.assertEqual(result.solver_type, "clifford_qc_acase")
        self.assertAlmostEqual(result.energy_ev, exact.energy_ev, places=8)
        self.assertLess(result.residual, 1e-8)
        self.assertTrue(result.converged)
        for row, reference_row in zip(result.one_rdm, exact.one_rdm):
            for value, reference in zip(row, reference_row):
                self.assertAlmostEqual(value, reference, places=7)

    def test_truncated_growth_stays_a_variational_upper_bound(self):
        ham = hubbard_chain(4, 4.0)
        exact = ExactDiagonalizationSolver().solve(ham)
        result = CliffordQCACASESolver(max_basis_size=6).solve(ham)

        self.assertGreater(result.energy_ev, exact.energy_ev)
        self.assertLessEqual(result.metadata["basis_size"], 6)
        self.assertFalse(result.converged)
        self.assertGreater(result.residual, 1e-8)

    def test_particle_number_is_conserved_by_the_projected_rdm(self):
        ham = hubbard_chain(4, 4.0)
        result = CliffordQCACASESolver(max_basis_size=8).solve(ham)
        trace = sum(result.one_rdm[i][i] for i in range(result.n_orbitals))
        self.assertAlmostEqual(trace, 4.0, places=6)
        self.assertAlmostEqual(sum(result.natural_occupations), 4.0, places=6)

    def test_total_energy_splits_into_electronic_plus_constant(self):
        ham = hubbard_chain(2, 2.0, u=2.0)
        ham.constant = 5.0  # eV
        result = CliffordQCACASESolver(max_basis_size=8).solve(ham)
        self.assertAlmostEqual(result.constant_energy_ev, 5.0, places=8)
        self.assertAlmostEqual(
            result.energy_ev, result.electronic_energy_ev + result.constant_energy_ev, places=8
        )


@unittest.skipUnless(HAS_SUBSPACE, "clifford_qc.subspace optional sibling not installed")
class TestACASEReporting(unittest.TestCase):
    def test_growth_diagnostics_are_not_disguised_as_adapt_quantities(self):
        """A Ritz growth has no operator gradients and no variational parameters."""
        result = CliffordQCACASESolver(max_basis_size=6).solve(hubbard_chain(4, 4.0))
        self.assertEqual(result.operator_gradients, [])
        self.assertEqual(result.operator_parameters, [])
        self.assertEqual(result.residual_kind, "ritz_residual_norm")
        self.assertEqual(result.metadata["residual_unit"], "Hartree")
        self.assertNotIn("residual_gradient", result.metadata)
        self.assertTrue(result.metadata["predicted_lowerings"])
        self.assertEqual(result.metadata["backend_api"], "clifford_qc.subspace.run_acase")

    def test_convergence_is_fail_closed_without_a_residual(self):
        """No residual means unconverged unless the candidate pool was exhausted."""
        result = CliffordQCACASESolver(
            max_basis_size=4, compute_ritz_residual=False
        ).solve(hubbard_chain(4, 4.0))
        self.assertIsNone(result.residual)
        self.assertEqual(result.residual_kind, "")
        self.assertFalse(result.converged)

    def test_a_closed_shell_sector_is_refused_with_its_reason(self):
        ham = hubbard_chain(2, 4.0, u=2.0)
        with self.assertRaisesRegex(ValueError, "no symmetry-preserving determinant excitations"):
            CliffordQCACASESolver().solve(ham)

    def test_non_hermitian_input_is_refused(self):
        ham = hubbard_chain(2, 2.0)
        ham.h1[0][1] += 0.5
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            CliffordQCACASESolver().solve(ham)


@unittest.skipUnless(HAS_SUBSPACE, "clifford_qc.subspace optional sibling not installed")
class TestADAPTWarmStartedACASE(unittest.TestCase):
    """The composition, and what it refuses to claim.

    A converged ADAPT state is stationary against the same excitation family
    A-CASE grows with, so the subspace can decline to grow while still sitting
    above the exact energy.  That is a property of the composition, not a bug,
    and the solver must not read "no growth" as "converged".
    """

    def test_the_adapt_reference_is_recorded_in_the_result(self):
        result = CliffordQCADAPTACASESolver(
            adapt_max_operators=2, max_basis_size=6
        ).solve(hubbard_chain(2, 2.0, u=2.0))

        self.assertEqual(result.solver_type, "clifford_qc_adapt_acase")
        self.assertEqual(result.metadata["reference"], "adapt_vqe_state")
        self.assertEqual(result.metadata["warm_start_api"], "clifford_qc.subspace.adapt_warm_start")
        self.assertTrue(result.metadata["adapt_operators"])
        self.assertIn("adapt_energy_ev", result.metadata)

    def test_convergence_follows_the_residual_not_the_absence_of_growth(self):
        solver = CliffordQCADAPTACASESolver(
            adapt_max_operators=2, max_basis_size=6, residual_threshold=1e-6
        )
        result = solver.solve(hubbard_chain(2, 2.0, u=2.0))
        self.assertEqual(result.converged, result.residual < 1e-6)
        if not result.metadata["grew_beyond_reference"]:
            # Stalled at the reference: the energy must then be the reference's.
            self.assertAlmostEqual(
                result.energy_ev, result.metadata["reference_energy_ev"], places=8
            )

    def test_warm_start_is_still_a_variational_upper_bound(self):
        ham = hubbard_chain(4, 4.0)
        exact = ExactDiagonalizationSolver().solve(ham)
        result = CliffordQCADAPTACASESolver(
            adapt_max_operators=2, max_basis_size=8
        ).solve(ham)
        self.assertGreater(result.energy_ev, exact.energy_ev - 1e-8)

    def test_the_factory_builds_the_composed_solver(self):
        solver = create_quantum_solver("adapt_warm_start", adapt_max_operators=1)
        self.assertIsInstance(solver, CliffordQCADAPTACASESolver)
        self.assertEqual(solver.adapt_max_operators, 1)


@unittest.skipUnless(HAS_SUBSPACE, "clifford_qc.subspace optional sibling not installed")
class TestOptionalDependencyBoundary(unittest.TestCase):
    """A-CASE must not inherit ADAPT's chemistry extra.

    The excitation pool is the only part of the backend that needs
    ``openfermion``, and A-CASE never uses it -- loading the ADAPT surface to
    reach the model/state helpers made the extra a hard dependency of a method
    that does not touch it.
    """

    SCRIPT = textwrap.dedent(
        """
        import builtins, sys
        real_import = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name == "openfermion" or name.startswith("openfermion."):
                raise ImportError("No module named 'openfermion' (simulated)")
            return real_import(name, *args, **kwargs)
        builtins.__import__ = blocked
        for module in [m for m in sys.modules if "openfermion" in m or "clifford_qc" in m]:
            del sys.modules[module]

        from qeanalyzer.quantum import (build_hubbard_hamiltonian, clifford_qc_available,
                                        clifford_qc_subspace_available, create_quantum_solver)
        ham = build_hubbard_hamiltonian(
            n_orbitals=2, n_electrons=2.0,
            hopping_t={(0, 1): 1.0, (1, 0): 1.0}, onsite_u=2.0)
        print("adapt_available", clifford_qc_available())
        print("acase_available", clifford_qc_subspace_available())
        print("acase_energy", create_quantum_solver("acase", max_basis_size=8).solve(ham).energy_ev)
        try:
            create_quantum_solver("adapt_vqe").solve(ham)
        except ImportError:
            print("adapt_raised ImportError")
        """
    )

    def test_acase_runs_without_the_chemistry_extra(self):
        completed = subprocess.run(
            [sys.executable, "-c", self.SCRIPT], capture_output=True, text=True, timeout=300
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout
        self.assertIn("adapt_available False", output)
        self.assertIn("acase_available True", output)
        self.assertIn("adapt_raised ImportError", output)
        energy = float(output.split("acase_energy")[1].split()[0])
        exact = ExactDiagonalizationSolver().solve(hubbard_chain(2, 2.0, u=2.0)).energy_ev
        self.assertAlmostEqual(energy, exact, places=8)


class TestACASEWithoutTheBackend(unittest.TestCase):
    """The name resolves whether or not the optional sibling is installed."""

    def test_the_solver_object_exists_without_clifford_qc(self):
        self.assertIsInstance(create_quantum_solver("acase"), CliffordQCACASESolver)

    @unittest.skipIf(HAS_SUBSPACE, "clifford_qc.subspace is installed here")
    def test_a_missing_backend_names_what_to_install(self):
        with self.assertRaisesRegex(ImportError, "clifford_qc"):
            CliffordQCACASESolver().solve(hubbard_chain(2, 2.0))


if __name__ == "__main__":
    unittest.main()
