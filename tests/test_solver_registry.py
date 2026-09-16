"""The solver registry is the one place a method name is resolved.

Before it existed, adding a method meant editing an if-chain and three alias
frozensets, and the only way to ask "what can this workflow drive?" was to read
the factory.
"""

from __future__ import annotations

import unittest

from qeanalyzer.quantum import (
    CliffordQCACASESolver,
    CliffordQCADAPTACASESolver,
    CliffordQCADAPTSolver,
    ComparisonSolver,
    ExactDiagonalizationSolver,
    SolverSpec,
    available_solvers,
    create_quantum_solver,
    describe_solvers,
    normalize_solver_type,
    register_solver,
    solver_names,
    unregister_solver,
)
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult, QuantumSolver
from qeanalyzer.quantum.solver_api import ACASE_ALIASES, ADAPT_ALIASES, EXACT_ALIASES


class _NullSolver(QuantumSolver):
    def solve(self, hamiltonian, active_space=None, initial_state=None, **kwargs):
        return QuantumRunResult(
            energy_ev=0.0, electronic_energy_ev=0.0, solver_type="null",
            n_orbitals=hamiltonian.n_orbitals, n_electrons=hamiltonian.n_electrons,
            n_spin_orbitals=hamiltonian.n_spin_orbitals,
        )


class TestRegistryContents(unittest.TestCase):
    def test_every_bridged_method_is_reachable_by_name(self):
        expected = {
            "exact": ExactDiagonalizationSolver,
            "adapt_vqe": CliffordQCADAPTSolver,
            "acase": CliffordQCACASESolver,
            "adapt_acase": CliffordQCADAPTACASESolver,
            "compare": ComparisonSolver,
        }
        for name, cls in expected.items():
            with self.subTest(name=name):
                self.assertIsInstance(create_quantum_solver(name), cls)
        self.assertEqual(set(solver_names()) & set(expected), set(expected))

    def test_names_are_normalized_across_case_and_separator(self):
        for alias in ("a-case", "A_CASE", "acase", "ACase", " acase "):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_solver_type(alias), "acase")

    def test_alias_sets_are_derived_from_the_registry(self):
        """Hand-maintained copies were how the two old factories disagreed."""
        self.assertIn("fci", EXACT_ALIASES)
        self.assertIn("adapt", ADAPT_ALIASES)
        self.assertIn("subspace", ACASE_ALIASES)
        self.assertFalse(ADAPT_ALIASES & ACASE_ALIASES)

    def test_every_spec_declares_a_scientific_status(self):
        for spec in available_solvers():
            with self.subTest(name=spec.name):
                self.assertTrue(spec.scientific_status)
                self.assertTrue(spec.summary)

    def test_a_clifford_backed_method_says_what_it_needs(self):
        by_name = {spec.name: spec for spec in available_solvers()}
        self.assertEqual(by_name["adapt_vqe"].requires, "clifford_qc")
        self.assertEqual(by_name["acase"].requires, "clifford_qc.subspace")
        self.assertEqual(by_name["exact"].requires, "")

    def test_describe_lists_each_method_once(self):
        text = describe_solvers()
        for spec in available_solvers():
            with self.subTest(name=spec.name):
                self.assertIn(spec.name, text)
                self.assertIn(spec.scientific_status, text)


class TestRegistryErrors(unittest.TestCase):
    def test_unknown_name_lists_what_is_available(self):
        with self.assertRaises(ValueError) as caught:
            create_quantum_solver("dmrg")
        self.assertIn("acase", str(caught.exception))

    def test_ambiguous_name_is_refused_rather_than_guessed(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            normalize_solver_type("vqe")

    def test_exact_solver_still_rejects_options(self):
        with self.assertRaisesRegex(TypeError, "takes no options"):
            create_quantum_solver("exact", max_basis_size=4)

    def test_options_reach_each_backend_constructor(self):
        acase = create_quantum_solver("acase", max_basis_size=6, max_excitation_rank=1)
        self.assertEqual((acase.max_basis_size, acase.max_excitation_rank), (6, 1))
        warm = create_quantum_solver("adapt_acase", adapt_max_operators=3, max_basis_size=5)
        self.assertEqual((warm.adapt_max_operators, warm.max_basis_size), (3, 5))


class TestRuntimeRegistration(unittest.TestCase):
    SPEC = SolverSpec(
        name="test_null",
        factory=_NullSolver,
        summary="test double",
        scientific_status="test_double",
        aliases=frozenset({"test_null_alias"}),
    )

    def tearDown(self):
        try:
            unregister_solver("test_null")
        except KeyError:
            pass

    def test_a_registered_method_is_creatable_by_name_and_alias(self):
        register_solver(self.SPEC)
        self.assertIsInstance(create_quantum_solver("test_null"), _NullSolver)
        self.assertIsInstance(create_quantum_solver("test_null_alias"), _NullSolver)
        self.assertIn("test_null", solver_names())

    def test_registration_cannot_shadow_an_existing_name(self):
        shadow = SolverSpec(name="acase", factory=_NullSolver, summary="x",
                            scientific_status="test_double")
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_solver(shadow)
        self.assertIsInstance(create_quantum_solver("acase"), CliffordQCACASESolver)

    def test_a_reserved_ambiguous_name_cannot_be_claimed(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            register_solver(SolverSpec(name="vqe", factory=_NullSolver, summary="x",
                                       scientific_status="test_double"))

    def test_built_in_methods_cannot_be_unregistered(self):
        with self.assertRaisesRegex(ValueError, "built-in"):
            unregister_solver("exact")


if __name__ == "__main__":
    unittest.main()
