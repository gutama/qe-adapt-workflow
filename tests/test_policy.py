"""Tests for workflow policies, decision engine, and recovery rules."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io import parse_pw_input, read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import QERunResult, QERunStatus, build_run_result
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

_RELAX_INPUT = (
    "&CONTROL\n  calculation = 'relax'\n  prefix = 'si'\n/\n"
    "&SYSTEM\n  ibrav = 2\n  celldm(1) = 10.2\n  nat = 2\n  ntyp = 1\n  ecutwfc = 30.0\n/\n"
    "&ELECTRONS\n  conv_thr = 1.0d-6\n/\n"
    "&IONS\n/\n"
    "ATOMIC_SPECIES\n Si 28.086 Si.upf\n"
    "ATOMIC_POSITIONS (crystal)\n Si 0.0 0.0 0.0\n Si 0.25 0.25 0.25\n"
    "K_POINTS automatic\n 4 4 4 0 0 0\n"
)


class TestGeometryConvergenceIsNotSCFConvergence(unittest.TestCase):
    """A relaxation whose BFGS never converged must not be promoted to production SCF.

    Electronic convergence of the final ionic step says nothing about whether the
    geometry reached its force threshold.
    """

    @staticmethod
    def _relax_result(opt_converged: bool | None) -> QERunResult:
        return QERunResult(
            run_id="relax-01",
            calculation="relax",
            status=QERunStatus(
                completed=True,
                scf_converged=True,
                opt_converged=opt_converged,
                exit_status="converged",
            ),
        )

    def test_policy_fires_on_converged_geometry(self):
        self.assertTrue(RelaxToSCFPolicy().can_handle(self._relax_result(True)))

    def test_policy_does_not_fire_on_unconverged_geometry(self):
        self.assertFalse(RelaxToSCFPolicy().can_handle(self._relax_result(False)))

    def test_policy_does_not_fire_when_geometry_status_unknown(self):
        self.assertFalse(RelaxToSCFPolicy().can_handle(self._relax_result(None)))

    def test_unconverged_relax_keeps_false_opt_status(self):
        """build_run_result must not erase a False bfgs_converged into None."""
        result = build_run_result(
            pw_in=parse_pw_input(_RELAX_INPUT),
            pw_out=read_pw_output(FIXTURES / "relax_unconverged.out"),
            run_id="relax-unconv",
        )
        self.assertEqual(result.calculation, "relax")
        self.assertTrue(result.status.scf_converged)
        self.assertIs(result.status.opt_converged, False)
        self.assertFalse(RelaxToSCFPolicy().can_handle(result))

    def test_scf_run_has_no_geometry_status(self):
        """opt_converged stays None for calculations that optimize no geometry."""
        result = build_run_result(
            pw_in=read_pw_input(FIXTURES / "si_scf.in"),
            pw_out=read_pw_output(FIXTURES / "si_scf.out"),
            run_id="si-scf",
        )
        self.assertIsNone(result.status.opt_converged)


class TestHardErrorsAreNotRestarted(unittest.TestCase):
    """A run that aborted with a QE error must not be blindly restarted.

    InterruptedRecoveryPolicy is registered first, so before this fix it claimed
    every incomplete run and returned a bare restart_mode='restart' that just
    reproduced the same failure.
    """

    def setUp(self):
        self.result = build_run_result(
            pw_in=read_pw_input(FIXTURES / "si_scf.in"),
            pw_out=read_pw_output(FIXTURES / "scf_error.out"),
            run_id="scf-error",
        )

    def test_fixture_is_a_hard_error(self):
        self.assertEqual(self.result.status.exit_status, "error")
        self.assertFalse(self.result.status.completed)
        self.assertTrue(self.result.status.errors)

    def test_interrupted_policy_declines_hard_errors(self):
        self.assertFalse(InterruptedRecoveryPolicy().can_handle(self.result))

    def test_corrective_policy_claims_the_run_instead(self):
        selected = default_registry.select_policy(self.result)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.name, "unconverged_scf_recovery")

    def test_recovery_actually_changes_scf_settings(self):
        """The chosen policy must alter something, not just restart unchanged."""
        decision = default_registry.select_policy(self.result).evaluate(
            self.result, prev_input=read_pw_input(FIXTURES / "si_scf.in")
        )
        self.assertNotEqual(decision.modified_namelists, {"CONTROL": {"restart_mode": "restart"}})
        self.assertTrue(decision.modified_namelists)

    def test_genuine_interruption_still_restarts(self):
        interrupted = build_run_result(
            pw_out=read_pw_output(FIXTURES / "interrupted.out"), run_id="interrupted"
        )
        self.assertEqual(interrupted.status.exit_status, "interrupted")
        self.assertTrue(InterruptedRecoveryPolicy().can_handle(interrupted))


if __name__ == "__main__":
    unittest.main()
