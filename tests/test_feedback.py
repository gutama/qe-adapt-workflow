"""Tests that quantum-to-DFT policies are deterministic and explicitly heuristic."""

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
    def setUp(self):
        self.pw_in = read_pw_input(FIXTURES / "si_scf.in")
        self.result = build_run_result(
            pw_in=self.pw_in,
            pw_out=read_pw_output(FIXTURES / "si_scf.out"),
            qe_xml=read_qe_xml(FIXTURES / "si_scf.xml"),
            run_id="dft-01",
        )

    @staticmethod
    def _result(**kwargs):
        defaults = dict(
            energy_ev=-10.0,
            electronic_energy_ev=-20.0,
            solver_type="exact_diagonalization",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
        )
        defaults.update(kwargs)
        return QuantumRunResult(**defaults)

    def _assert_experimental(self, decision):
        self.assertEqual(decision.metadata["scientific_status"], "experimental_heuristic")
        self.assertFalse(decision.metadata["validated_physical_self_consistency"])
        self.assertIn("EXPERIMENTAL HEURISTIC", decision.reason)

    def test_occupation_feedback(self):
        decision = OccupationFeedbackPolicy(adjusted_degauss=0.02).evaluate_feedback(
            self.result,
            self._result(natural_occupations=[1.4, 0.6]),
            prev_input=self.pw_in,
        )
        self._assert_experimental(decision)
        self.assertEqual(decision.modified_namelists["SYSTEM"]["occupations"], "smearing")
        self.assertEqual(decision.next_input.get_param("SYSTEM", "degauss"), 0.02)
        self.assertIn("not an RDM embedding update", decision.reason)

    def test_active_space_feedback(self):
        current = self.result.electronic.n_bands or 8
        decision = ActiveSpaceFeedbackPolicy(band_increment=4).evaluate_feedback(
            self.result,
            self._result(natural_occupations=[1.9, 0.1]),
            prev_input=self.pw_in,
        )
        self._assert_experimental(decision)
        self.assertEqual(decision.modified_namelists["SYSTEM"]["nbnd"], current + 4)

    def test_hubbard_feedback_is_legacy_heuristic(self):
        decision = HubbardUFeedbackPolicy(response_alpha=1.0).evaluate_feedback(
            self.result,
            self._result(one_rdm=[[1.4, 0.0], [0.0, 0.6]]),
            prev_input=self.pw_in,
        )
        self._assert_experimental(decision)
        self.assertEqual(decision.modified_namelists["SYSTEM"]["Hubbard_U(1)"], 0.4)
        self.assertIn("Legacy linear response heuristic", decision.reason)

    def test_dispatch(self):
        decision = apply_quantum_feedback(
            self.result,
            self._result(natural_occupations=[1.5, 0.5]),
            policy_name="occupation",
            prev_input=self.pw_in,
        )
        self.assertEqual(decision.policy_name, "occupation_feedback")
        with self.assertRaises(ValueError):
            apply_quantum_feedback(self.result, self._result(), policy_name="does-not-exist")


if __name__ == "__main__":
    unittest.main()
