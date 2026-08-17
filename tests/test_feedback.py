"""Tests for QuantumFeedbackPolicy implementations (src/qeanalyzer/quantum/feedback.py)."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult
from qeanalyzer.quantum.feedback import (
    ActiveSpaceFeedbackPolicy,
    HubbardUFeedbackPolicy,
    OccupationFeedbackPolicy,
    apply_quantum_feedback,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestQuantumFeedbackPolicies(unittest.TestCase):
    """Test quantum solver feedback policies and next-run parameter construction."""

    def setUp(self):
        self.pw_in = read_pw_input(FIXTURES / "si_scf.in")
        self.pw_out = read_pw_output(FIXTURES / "si_scf.out")
        self.xml = read_qe_xml(FIXTURES / "si_scf.xml")
        self.result = build_run_result(pw_in=self.pw_in, pw_out=self.pw_out, qe_xml=self.xml, run_id="dft-01")

    def test_occupation_feedback_multireference(self):
        # Correlated multireference result with fractional natural occupations
        q_res = QuantumRunResult(
            energy_ev=-10.0,
            electronic_energy_ev=-20.0,
            solver_type="exact_diagonalization",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            natural_occupations=[1.4, 0.6],  # Strongly fractional
        )

        policy = OccupationFeedbackPolicy(correlation_threshold=0.05, target_smearing="cold", adjusted_degauss=0.02)
        decision = policy.evaluate_feedback(self.result, q_res, prev_input=self.pw_in)

        self.assertEqual(decision.decision_type, "NEXT_RUN")
        self.assertEqual(decision.policy_name, "occupation_feedback")
        self.assertIn("SYSTEM", decision.modified_namelists)
        self.assertEqual(decision.modified_namelists["SYSTEM"]["occupations"], "smearing")
        self.assertEqual(decision.modified_namelists["SYSTEM"]["smearing"], "cold")
        self.assertEqual(decision.modified_namelists["SYSTEM"]["degauss"], 0.02)
        self.assertEqual(decision.modified_namelists["ELECTRONS"]["mixing_beta"], 0.3)

        self.assertIsNotNone(decision.next_input)
        self.assertEqual(decision.next_input.get_param("SYSTEM", "occupations"), "smearing")
        self.assertIn("smearing = 'cold'", decision.next_input_text)

    def test_occupation_feedback_single_reference(self):
        # Single-reference result with near-integer natural occupations
        q_res = QuantumRunResult(
            energy_ev=-10.0,
            electronic_energy_ev=-20.0,
            solver_type="exact_diagonalization",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            natural_occupations=[1.99, 0.01],
        )

        policy = OccupationFeedbackPolicy(correlation_threshold=0.05)
        decision = policy.evaluate_feedback(self.result, q_res, prev_input=self.pw_in)

        self.assertEqual(decision.decision_type, "NEXT_RUN")
        self.assertNotIn("SYSTEM", decision.modified_namelists)
        self.assertIn("single-reference", decision.reason)

    def test_active_space_feedback_expansion(self):
        # Boundary orbital significantly occupied -> triggers band expansion
        q_res = QuantumRunResult(
            energy_ev=-10.0,
            electronic_energy_ev=-20.0,
            solver_type="adapt_vqe",
            n_orbitals=4,
            n_electrons=4.0,
            n_spin_orbitals=8,
            natural_occupations=[1.9, 1.8, 0.2, 0.1],  # Boundary 0.1 > 0.02
        )

        policy = ActiveSpaceFeedbackPolicy(boundary_occupation_threshold=0.02, band_increment=4)
        decision = policy.evaluate_feedback(self.result, q_res, prev_input=self.pw_in)

        curr_nbnd = self.result.electronic.n_bands or 8
        self.assertEqual(decision.modified_namelists["SYSTEM"]["nbnd"], curr_nbnd + 4)
        self.assertIn("Expanding active space bands", decision.reason)
        self.assertIsNotNone(decision.next_input)
        self.assertEqual(decision.next_input.get_param("SYSTEM", "nbnd"), curr_nbnd + 4)

    def test_hubbard_u_feedback(self):
        q_res = QuantumRunResult(
            energy_ev=-10.0,
            electronic_energy_ev=-20.0,
            solver_type="exact_diagonalization",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            one_rdm=[[1.4, 0.1], [0.1, 0.6]],  # Local charge = 1.4, nominal = 1.0 -> delta_n = 0.4
        )

        policy = HubbardUFeedbackPolicy(response_alpha=1.0, species_index=1)
        decision = policy.evaluate_feedback(self.result, q_res, prev_input=self.pw_in)

        self.assertEqual(decision.modified_namelists["SYSTEM"]["lda_plus_u"], True)
        self.assertEqual(decision.modified_namelists["SYSTEM"]["Hubbard_U(1)"], 0.4)
        self.assertIn("Hubbard_U(1) to 0.4000 eV", decision.reason)

    def test_apply_quantum_feedback_convenience(self):
        q_res = QuantumRunResult(
            energy_ev=-10.0,
            electronic_energy_ev=-20.0,
            solver_type="adapt_vqe",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            natural_occupations=[1.5, 0.5],
        )

        dec = apply_quantum_feedback(self.result, q_res, policy_name="occupation", prev_input=self.pw_in)
        self.assertEqual(dec.policy_name, "occupation_feedback")

        with self.assertRaises(ValueError):
            apply_quantum_feedback(self.result, q_res, policy_name="nonexistent_policy")


if __name__ == "__main__":
    unittest.main()
