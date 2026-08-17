"""Tests for workflow policies, decision engine, and recovery rules."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import QERunResult, build_run_result
from qeanalyzer.workflow import (
    InterruptedRecoveryPolicy,
    NextRunDecision,
    PolicyRegistry,
    RelaxToSCFPolicy,
    SCFToNSCFPolicy,
    UnconvergedSCFPolicy,
    default_registry,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestWorkflowPolicies(unittest.TestCase):
    """Test policy rules and next-run generation."""

    def test_scf_to_nscf_policy(self):
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")
        result = build_run_result(pw_in=pw_in, pw_out=pw_out, qe_xml=qe_xml, run_id="si-scf-01")

        policy = SCFToNSCFPolicy()
        self.assertTrue(policy.can_handle(result))

        decision = policy.evaluate(result, prev_input=pw_in)
        self.assertEqual(decision.decision_type, "NEXT_RUN")
        self.assertFalse(decision.is_restart)
        self.assertEqual(decision.target_calculation, "nscf")
        self.assertIsNotNone(decision.next_input)
        self.assertEqual(decision.next_input.calculation, "nscf")
        self.assertGreaterEqual(decision.next_input.namelists["SYSTEM"]["nbnd"], 8)
        self.assertIn("calculation = 'nscf'", decision.next_input_text)

    def test_relax_to_scf_policy(self):
        pw_in = read_pw_input(FIXTURES / "si_vc_relax.in")
        pw_out = read_pw_output(FIXTURES / "si_vc_relax.out")
        qe_xml = read_qe_xml(FIXTURES / "si_vc_relax.xml")
        result = build_run_result(pw_in=pw_in, pw_out=pw_out, qe_xml=qe_xml, run_id="si-vcrelax-01")

        policy = RelaxToSCFPolicy()
        self.assertTrue(policy.can_handle(result))

        decision = policy.evaluate(result, prev_input=pw_in)
        self.assertEqual(decision.decision_type, "NEXT_RUN")
        self.assertFalse(decision.is_restart)
        self.assertEqual(decision.target_calculation, "scf")
        self.assertIsNotNone(decision.next_input)
        self.assertEqual(decision.next_input.calculation, "scf")
        self.assertEqual(decision.next_input.namelists["SYSTEM"]["ibrav"], 0)
        self.assertIsNotNone(decision.next_input.cell_parameters)

    def test_unconverged_scf_recovery(self):
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "failed_scf.out")
        result = build_run_result(pw_in=pw_in, pw_out=pw_out, run_id="failed-scf-01")

        policy = UnconvergedSCFPolicy()
        self.assertTrue(policy.can_handle(result))

        decision = policy.evaluate(result, prev_input=pw_in)
        self.assertEqual(decision.decision_type, "RETRY")
        self.assertFalse(decision.is_restart)
        self.assertEqual(decision.next_input.namelists["ELECTRONS"]["mixing_beta"], 0.3)
        self.assertEqual(decision.next_input.namelists["ELECTRONS"]["electron_maxstep"], 150)

    def test_interrupted_recovery(self):
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "interrupted.out")
        result = build_run_result(pw_in=pw_in, pw_out=pw_out, run_id="interrupted-01")

        policy = InterruptedRecoveryPolicy()
        self.assertTrue(policy.can_handle(result))

        decision = policy.evaluate(result, prev_input=pw_in)
        self.assertEqual(decision.decision_type, "RESTART")
        self.assertTrue(decision.is_restart)
        self.assertEqual(decision.next_input.namelists["CONTROL"]["restart_mode"], "restart")

    def test_default_registry_selection(self):
        pw_out_scf = read_pw_output(FIXTURES / "si_scf.out")
        result_scf = build_run_result(pw_out=pw_out_scf)
        matched_scf = default_registry.select_policy(result_scf)
        self.assertIsNotNone(matched_scf)
        self.assertEqual(matched_scf.name, "scf_to_nscf")

        pw_out_failed = read_pw_output(FIXTURES / "failed_scf.out")
        result_failed = build_run_result(pw_out=pw_out_failed)
        matched_failed = default_registry.select_policy(result_failed)
        self.assertIsNotNone(matched_failed)
        self.assertEqual(matched_failed.name, "unconverged_scf_recovery")

        pw_out_interrupted = read_pw_output(FIXTURES / "interrupted.out")
        result_interrupted = build_run_result(pw_out=pw_out_interrupted)
        matched_interrupted = default_registry.select_policy(result_interrupted)
        self.assertIsNotNone(matched_interrupted)
        self.assertEqual(matched_interrupted.name, "interrupted_calculation_restart")

    def test_decision_to_dict(self):
        decision = NextRunDecision(
            decision_type="NEXT_RUN",
            policy_name="scf_to_nscf",
            policy_version="1.0",
            reason="Advancing to NSCF",
            target_calculation="nscf",
            is_restart=False,
            parent_run="run-01",
        )
        d = decision.to_dict()
        self.assertEqual(d["decision_type"], "NEXT_RUN")
        self.assertEqual(d["policy"]["name"], "scf_to_nscf")
        self.assertFalse(d["is_restart"])


if __name__ == "__main__":
    unittest.main()
