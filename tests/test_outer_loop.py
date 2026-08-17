"""Tests for outer-loop convergence checking, ledger tracking, and controller (src/qeanalyzer/workflow/outer_loop.py)."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult
from qeanalyzer.workflow.outer_loop import (
    ConvergenceCriteria,
    OuterLoopController,
    OuterLoopIterationRecord,
    OuterLoopLedger,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestOuterLoopConvergence(unittest.TestCase):
    """Test outer-loop multi-criteria convergence evaluation and ledger tracking."""

    def setUp(self):
        self.pw_in = read_pw_input(FIXTURES / "si_scf.in")
        self.pw_out = read_pw_output(FIXTURES / "si_scf.out")
        self.xml = read_qe_xml(FIXTURES / "si_scf.xml")
        self.dft_res = build_run_result(pw_in=self.pw_in, pw_out=self.pw_out, qe_xml=self.xml, run_id="dft_001")

    def test_criteria_serialization(self):
        crit = ConvergenceCriteria(energy_tolerance_ev=1e-5, rdm_tolerance=1e-4, max_outer_iterations=10)
        d = crit.to_dict()
        rec = ConvergenceCriteria.from_dict(d)
        self.assertEqual(rec.energy_tolerance_ev, 1e-5)
        self.assertEqual(rec.rdm_tolerance, 1e-4)
        self.assertEqual(rec.max_outer_iterations, 10)

    def test_outer_loop_ledger_convergence(self):
        ledger = OuterLoopLedger(criteria=ConvergenceCriteria(energy_tolerance_ev=1e-4, rdm_tolerance=1e-3))

        # Iteration 1
        q_res1 = QuantumRunResult(
            energy_ev=-15.820,
            electronic_energy_ev=-25.820,
            solver_type="adapt_vqe",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            one_rdm=[[1.2, 0.1], [0.1, 0.8]],
            operator_gradients=[0.005],
        )
        ledger.record_iteration(self.dft_res, q_res1)
        check1 = ledger.check_convergence()
        self.assertFalse(check1.is_converged)
        self.assertEqual(check1.iteration, 1)

        # Iteration 2 (converged delta E = 3e-5 eV, small delta rdm = 2e-4)
        q_res2 = QuantumRunResult(
            energy_ev=-15.82003,
            electronic_energy_ev=-25.82003,
            solver_type="adapt_vqe",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            one_rdm=[[1.2001, 0.1001], [0.1001, 0.7999]],
            operator_gradients=[0.0002],
        )
        ledger.record_iteration(self.dft_res, q_res2)
        check2 = ledger.check_convergence()
        self.assertTrue(check2.is_converged)
        self.assertEqual(check2.iteration, 2)
        self.assertTrue(check2.passed_criteria["energy"])
        self.assertTrue(check2.passed_criteria["rdm"])
        self.assertTrue(check2.passed_criteria["gradient"])
        self.assertIn("CONVERGED", check2.summary())

    @staticmethod
    def _quantum_result(energy_ev: float, diag: float = 1.0) -> QuantumRunResult:
        return QuantumRunResult(
            energy_ev=energy_ev,
            electronic_energy_ev=energy_ev - 10.0,
            solver_type="adapt_vqe",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            one_rdm=[[diag, 0.0], [0.0, diag]],
        )

    def test_outer_loop_max_iterations(self):
        """Hitting the limit without meeting the criteria is reported as such."""
        ledger = OuterLoopLedger(criteria=ConvergenceCriteria(max_outer_iterations=2))
        ledger.record_iteration(self.dft_res, self._quantum_result(-10.0, diag=1.0))
        ledger.record_iteration(self.dft_res, self._quantum_result(-14.0, diag=0.4))

        check = ledger.check_convergence()
        self.assertFalse(check.is_converged)
        self.assertIn("Reached maximum", check.reason)

    def test_convergence_on_final_allowed_iteration_is_converged(self):
        """Meeting every criterion on the last allowed iteration is a success.

        The iteration limit must be evaluated after the criteria, otherwise a
        genuinely self-consistent run is recorded as having merely run out.
        """
        ledger = OuterLoopLedger(criteria=ConvergenceCriteria(max_outer_iterations=2))
        q_res = self._quantum_result(-10.0)
        ledger.record_iteration(self.dft_res, q_res)
        ledger.record_iteration(self.dft_res, q_res)

        check = ledger.check_convergence()
        self.assertTrue(check.is_converged)
        self.assertNotIn("Reached maximum", check.reason)
        self.assertTrue(all(check.passed_criteria.values()))

    def test_outer_loop_controller_orchestration(self):
        controller = OuterLoopController(
            criteria=ConvergenceCriteria(energy_tolerance_ev=1e-4),
            feedback_policy="occupation_feedback",
        )

        q_res1 = QuantumRunResult(
            energy_ev=-15.0,
            electronic_energy_ev=-25.0,
            solver_type="adapt_vqe",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            natural_occupations=[1.4, 0.6],
            one_rdm=[[1.4, 0.0], [0.0, 0.6]],
        )

        # Step 1 -> Should apply feedback and request next DFT run
        dec1 = controller.step(self.dft_res, q_res1, prev_input=self.pw_in)
        self.assertEqual(dec1.decision_type, "NEXT_RUN")
        self.assertEqual(dec1.policy_name, "occupation_feedback")
        self.assertIn("smearing", dec1.modified_namelists["SYSTEM"]["occupations"])

        # Step 2 -> Highly converged result -> Should STOP with outer-loop convergence
        q_res2 = QuantumRunResult(
            energy_ev=-15.00002,
            electronic_energy_ev=-25.00002,
            solver_type="adapt_vqe",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            natural_occupations=[1.4, 0.6],
            one_rdm=[[1.4, 0.0], [0.0, 0.6]],
            operator_gradients=[0.0001],
        )
        dec2 = controller.step(self.dft_res, q_res2, prev_input=self.pw_in)
        self.assertEqual(dec2.decision_type, "STOP")
        self.assertIn("Outer Loop Converged", dec2.reason)

    def test_ledger_serialization_roundtrip(self):
        ledger = OuterLoopLedger()
        q_res = QuantumRunResult(
            energy_ev=-12.34,
            electronic_energy_ev=-22.34,
            solver_type="exact_diagonalization",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
        )
        ledger.record_iteration(self.dft_res, q_res)

        d = ledger.to_dict()
        reconstructed = OuterLoopLedger.from_dict(d)
        self.assertEqual(len(reconstructed.iterations), 1)
        self.assertAlmostEqual(reconstructed.iterations[0].quantum_energy_ev, -12.34)


if __name__ == "__main__":
    unittest.main()
