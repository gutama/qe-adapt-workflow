"""Fail-closed outer-loop convergence tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult
from qeanalyzer.workflow.outer_loop import ConvergenceCriteria, OuterLoopController, OuterLoopLedger

FIXTURES = Path(__file__).parent / "fixtures"


class TestOuterLoop(unittest.TestCase):
    def setUp(self):
        self.pw_in = read_pw_input(FIXTURES / "si_scf.in")
        self.dft = build_run_result(
            pw_in=self.pw_in,
            pw_out=read_pw_output(FIXTURES / "si_scf.out"),
            qe_xml=read_qe_xml(FIXTURES / "si_scf.xml"),
            run_id="dft-1",
        )

    @staticmethod
    def q(energy, rdm=None, residual=None):
        metadata = {}
        if residual is not None:
            metadata["residual_gradient"] = residual
        return QuantumRunResult(
            energy_ev=energy,
            electronic_energy_ev=energy,
            solver_type="clifford_qc_adapt_vqe",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            one_rdm=rdm or [],
            metadata=metadata,
            natural_occupations=[1.5, 0.5],
        )

    def test_convergence_on_final_allowed_iteration_is_converged(self):
        """Meeting every criterion on the last allowed iteration is a success.

        The iteration limit must be evaluated after the criteria, otherwise a
        genuinely self-consistent run is recorded as having merely run out.
        """
        ledger = OuterLoopLedger(ConvergenceCriteria(max_outer_iterations=2))
        rdm = [[1.0, 0.0], [0.0, 1.0]]
        ledger.record_iteration(self.dft, self.q(-10.0, rdm=rdm, residual=1e-6))
        ledger.record_iteration(self.dft, self.q(-10.0, rdm=rdm, residual=1e-6))

        check = ledger.check_convergence()
        self.assertTrue(check.is_converged)
        self.assertNotIn("Reached maximum", check.reason)
        self.assertTrue(all(v is not False for v in check.passed_criteria.values()))

    def test_criteria_roundtrip_includes_required_flags(self):
        c = ConvergenceCriteria(require_rdm=False, require_gradient=True)
        r = ConvergenceCriteria.from_dict(c.to_dict())
        self.assertFalse(r.require_rdm)
        self.assertTrue(r.require_gradient)

    def test_missing_required_quantities_do_not_pass(self):
        ledger = OuterLoopLedger()
        ledger.record_iteration(self.dft, self.q(-10.0))
        ledger.record_iteration(self.dft, self.q(-10.0))
        result = ledger.check_convergence()
        self.assertFalse(result.is_converged)
        self.assertFalse(result.passed_criteria["rdm"])
        self.assertFalse(result.passed_criteria["gradient"])

    def test_all_available_criteria_can_converge(self):
        ledger = OuterLoopLedger(ConvergenceCriteria(
            energy_tolerance_ev=1e-4,
            rdm_tolerance=1e-3,
            gradient_tolerance=1e-3,
        ))
        ledger.record_iteration(self.dft, self.q(-10.0, [[1.0, 0.0], [0.0, 1.0]], 5e-3))
        ledger.record_iteration(self.dft, self.q(-10.00001, [[1.0001, 0.0], [0.0, 0.9999]], 2e-4))
        result = ledger.check_convergence()
        self.assertTrue(result.is_converged)
        self.assertTrue(all(v is True for v in result.passed_criteria.values()))

    def test_optional_gradient_can_be_disabled_for_plumbing_mock(self):
        ledger = OuterLoopLedger(ConvergenceCriteria(require_gradient=False))
        rdm = [[1.0, 0.0], [0.0, 1.0]]
        ledger.record_iteration(self.dft, self.q(-10.0, rdm))
        ledger.record_iteration(self.dft, self.q(-10.0, rdm))
        result = ledger.check_convergence()
        self.assertTrue(result.is_converged)
        self.assertIsNone(result.passed_criteria["gradient"])

    def test_controller_does_not_stop_on_missing_gradient(self):
        controller = OuterLoopController(criteria=ConvergenceCriteria(), feedback_policy="occupation")
        rdm = [[1.0, 0.0], [0.0, 1.0]]
        controller.step(self.dft, self.q(-10.0, rdm), prev_input=self.pw_in)
        decision = controller.step(self.dft, self.q(-10.0, rdm), prev_input=self.pw_in)
        self.assertEqual(decision.decision_type, "NEXT_RUN")


if __name__ == "__main__":
    unittest.main()
