"""Multi-arm comparison tests.

A comparison record is evidence about several methods on *one* Hamiltonian, so
the parts that could manufacture a wrong conclusion are what these tests pin:
which arms may be ranked, that every arm answered the same sector, and that a
declared budget is never invented by the runner.
"""

from __future__ import annotations

import unittest
import warnings

from qeanalyzer.quantum import (
    ComparisonSolver,
    ExactDiagonalizationSolver,
    SolverArm,
    build_hubbard_hamiltonian,
    clifford_qc_subspace_available,
    compare_solvers,
    create_quantum_solver,
)
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult, QuantumSolver
from qeanalyzer.quantum.comparison import coerce_arm, hamiltonian_fingerprint
from qeanalyzer.quantum.solver_api import SolverSpec, register_solver, unregister_solver

HAS_SUBSPACE = clifford_qc_subspace_available()


def dimer(u: float = 2.0):
    return build_hubbard_hamiltonian(
        n_orbitals=2, n_electrons=2.0, hopping_t={(0, 1): 1.0, (1, 0): 1.0}, onsite_u=u
    )


class _WrongSectorSolver(QuantumSolver):
    """Answers a different electron count than it was handed."""

    def solve(self, hamiltonian, active_space=None, initial_state=None, **kwargs):
        return QuantumRunResult(
            energy_ev=-99.0, electronic_energy_ev=-99.0, solver_type="wrong_sector",
            n_orbitals=hamiltonian.n_orbitals, n_electrons=hamiltonian.n_electrons + 2,
            n_spin_orbitals=hamiltonian.n_spin_orbitals,
        )


class _BrokenSolver(QuantumSolver):
    def solve(self, hamiltonian, active_space=None, initial_state=None, **kwargs):
        raise RuntimeError("backend exploded")


class TestArmSpecification(unittest.TestCase):
    def test_arms_can_be_named_paired_or_described(self):
        self.assertEqual(coerce_arm("acase").solver_type, "acase")
        self.assertEqual(coerce_arm(("acase", {"max_basis_size": 4})).options,
                         {"max_basis_size": 4})
        spec = coerce_arm({"solver_type": "acase", "label": "m4",
                           "options": {"max_basis_size": 4}})
        self.assertEqual((spec.name, spec.options), ("m4", {"max_basis_size": 4}))
        self.assertEqual(coerce_arm(SolverArm("exact")).name, "exact")

    def test_duplicate_labels_are_refused(self):
        """Two arms under one name would overwrite each other in the record."""
        with self.assertRaisesRegex(ValueError, "unique"):
            compare_solvers(dimer(), [
                {"solver_type": "exact", "label": "arm"},
                {"solver_type": "acase", "label": "arm"},
            ])

    def test_one_method_may_appear_twice_under_different_labels(self):
        """Comparing two budgets of the same method is the normal case."""
        comparison = compare_solvers(dimer(), [
            {"solver_type": "exact", "label": "exact_a"},
            {"solver_type": "fci", "label": "exact_b"},
        ])
        self.assertEqual([arm.label for arm in comparison.arms], ["exact_a", "exact_b"])

    def test_an_empty_comparison_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one arm"):
            compare_solvers(dimer(), [])


class TestComparisonRecord(unittest.TestCase):
    def test_every_arm_is_recorded_against_one_hamiltonian(self):
        ham = dimer()
        comparison = compare_solvers(ham, ["exact"])
        self.assertEqual(comparison.hamiltonian_sha256, hamiltonian_fingerprint(ham))
        self.assertEqual(comparison.reference_energy_ev,
                         ExactDiagonalizationSolver().solve(ham).energy_ev)
        self.assertEqual(comparison.selected_label, "exact")

    def test_the_mock_is_recorded_but_never_ranked(self):
        """It is derived from the exact answer, so on energy it would often win."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            comparison = compare_solvers(dimer(), ["simulated_adapt"], on_error="record")
        arm = comparison.outcomes["simulated_adapt"]
        self.assertIsNotNone(arm.result)
        self.assertFalse(arm.rankable)
        self.assertIsNone(comparison.selected_label)
        self.assertEqual(arm.scientific_status, "workflow_mock")

    def test_an_explicit_selection_overrides_the_energy_rule(self):
        comparison = compare_solvers(dimer(), ["exact"], selection="exact")
        self.assertEqual(comparison.selection_rule, "exact")
        with self.assertRaisesRegex(ValueError, "neither 'lowest_energy'"):
            compare_solvers(dimer(), ["exact"], selection="not_an_arm")

    def test_a_failing_arm_can_be_recorded_instead_of_ending_the_run(self):
        register_solver(SolverSpec(name="test_broken", factory=_BrokenSolver,
                                   summary="raises", scientific_status="test_double"))
        try:
            with self.assertRaisesRegex(RuntimeError, "backend exploded"):
                compare_solvers(dimer(), ["exact", "test_broken"])
            comparison = compare_solvers(dimer(), ["exact", "test_broken"], on_error="record")
        finally:
            unregister_solver("test_broken")
        broken = comparison.outcomes["test_broken"]
        self.assertIsNone(broken.result)
        self.assertIn("backend exploded", broken.error)
        self.assertEqual(comparison.selected_label, "exact")

    def test_an_arm_that_changed_the_sector_is_refused(self):
        register_solver(SolverSpec(name="test_wrong_sector", factory=_WrongSectorSolver,
                                   summary="wrong sector", scientific_status="test_double"))
        try:
            with self.assertRaisesRegex(ValueError, "electrons"):
                compare_solvers(dimer(), ["exact", "test_wrong_sector"])
        finally:
            unregister_solver("test_wrong_sector")

    def test_budget_matching_is_only_ever_what_the_caller_declared(self):
        undeclared = compare_solvers(dimer(), ["exact"]).to_dict()
        self.assertFalse(undeclared["matched_budget_declared"])
        self.assertIn("NOT DECLARED", compare_solvers(dimer(), ["exact"]).summary())
        declared = compare_solvers(dimer(), ["exact"], budget_note="one arm").to_dict()
        self.assertTrue(declared["matched_budget_declared"])

    def test_the_record_is_compact_unless_full_results_are_requested(self):
        comparison = compare_solvers(dimer(), ["exact"])
        self.assertNotIn("result", comparison.to_dict()["arms"][0])
        self.assertIn("result", comparison.to_dict(include_results=True)["arms"][0])
        self.assertEqual(comparison.to_dict()["arms"][0]["error_vs_reference_ev"], 0.0)


class TestComparisonSolver(unittest.TestCase):
    def test_it_returns_the_selected_arm_with_the_record_attached(self):
        ham = dimer()
        result = ComparisonSolver(arms=["exact"], budget_note="single arm").solve(ham)
        exact = ExactDiagonalizationSolver().solve(ham)
        self.assertAlmostEqual(result.energy_ev, exact.energy_ev, places=10)
        self.assertEqual(result.metadata["selected_arm"], "exact")
        self.assertEqual(result.metadata["comparison"]["budget_note"], "single arm")
        self.assertEqual(result.metadata["comparison"]["scientific_status"], "comparison_record")

    def test_the_arm_result_is_not_mutated_by_the_composite(self):
        """Attaching the record in place would nest it inside one of its arms."""
        result = ComparisonSolver(arms=["exact"]).solve(dimer())
        arm_metadata = result.metadata["comparison"]["arms"][0]
        self.assertNotIn("comparison", arm_metadata.get("result", {}).get("metadata", {}))

    def test_no_rankable_arm_is_an_error_rather_than_a_silent_mock_result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(RuntimeError, "no comparison arm"):
                ComparisonSolver(arms=["simulated_adapt"]).solve(dimer())

    def test_it_is_reachable_through_the_registry(self):
        solver = create_quantum_solver("multi_arm", arms=["exact"])
        self.assertIsInstance(solver, ComparisonSolver)

    @unittest.skipUnless(HAS_SUBSPACE, "clifford_qc.subspace optional sibling not installed")
    def test_methods_are_ranked_as_variational_upper_bounds(self):
        comparison = compare_solvers(
            dimer(),
            [
                "exact",
                {"solver_type": "acase", "label": "acase_m3", "options": {"max_basis_size": 3}},
                {"solver_type": "acase", "label": "acase_m8", "options": {"max_basis_size": 8}},
            ],
            budget_note="retained basis size 3 vs 8; not cost-matched",
        )
        errors = {arm.label: comparison.error_vs_reference_ev(arm.label)
                  for arm in comparison.arms}
        self.assertAlmostEqual(errors["exact"], 0.0, places=10)
        self.assertGreaterEqual(errors["acase_m3"], -1e-9)
        self.assertLessEqual(errors["acase_m8"], errors["acase_m3"] + 1e-9)
        self.assertEqual(comparison.selected_label, "exact")


if __name__ == "__main__":
    unittest.main()
