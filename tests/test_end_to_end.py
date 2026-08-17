"""End-to-end workflow plumbing tests.

Scientific ADAPT correctness is owned by clifford_qc; this file tests QE/workflow
integration and uses the explicitly named simulated solver only where a local
closed-loop stub is useful.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.plotting import plot_scf_convergence
from qeanalyzer.quantum import (
    ExactDiagonalizationSolver,
    SimulatedADAPTVQESolver,
    apply_quantum_feedback,
    build_band_model_hamiltonian,
    parse_fcidump,
    select_active_space,
    write_fcidump,
)
from qeanalyzer.report import dump_result_json, generate_text_report
from qeanalyzer.workflow import ConvergenceCriteria, OuterLoopController, OuterLoopLedger, WorkflowLedger, plan_next_calculation

FIXTURES = Path(__file__).parent / "fixtures"
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


class TestEndToEndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw_in = read_pw_input(FIXTURES / "si_scf.in")
        cls.pw_out = read_pw_output(FIXTURES / "si_scf.out")
        cls.xml = read_qe_xml(FIXTURES / "si_scf.xml")
        cls.result = build_run_result(
            pw_in=cls.pw_in,
            pw_out=cls.pw_out,
            qe_xml=cls.xml,
            run_id="si_scf_001",
        )

    def test_stage_a_parse_report_plot_next(self):
        self.assertTrue(self.result.status.scf_converged)
        self.assertIn("Quantum ESPRESSO Run Analysis", generate_text_report(self.result, markdown=True))
        self.assertIn('"schema_version": "1.0"', dump_result_json(self.result))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            if HAS_MATPLOTLIB:
                plot_scf_convergence(self.result, output_path=root / "scf.png")
                self.assertTrue((root / "scf.png").exists())
            ledger = WorkflowLedger(workflow_id="wf")
            decision = plan_next_calculation(
                self.result,
                self.pw_in,
                output_dir=root / "002_nscf",
                ledger=ledger,
            )
            self.assertEqual(decision.policy_name, "scf_to_nscf")
            self.assertTrue((root / "002_nscf" / "pw.in").exists())

    def test_stage_b_band_model_is_plumbing_only_and_fci_runs(self):
        active = select_active_space(self.result, method="band_index", band_start=1, band_end=2)
        ham = build_band_model_hamiltonian(self.result, active, onsite_u_ev=2.0)
        self.assertFalse(ham.metadata["ab_initio"])
        q_result = ExactDiagonalizationSolver().solve(ham, active_space=active)
        self.assertTrue(q_result.converged)

    def test_stage_c_fcidump_is_hartree(self):
        active = select_active_space(self.result, method="band_index", band_start=1, band_end=2)
        ham = build_band_model_hamiltonian(self.result, active, onsite_u_ev=1.5)
        loaded = parse_fcidump(write_fcidump(ham))
        self.assertEqual(loaded.energy_unit, "Hartree")

    def test_stage_d_e_mock_loop_requires_explicit_missing_criteria_policy(self):
        active = select_active_space(self.result, method="band_index", band_start=1, band_end=2)
        ham = build_band_model_hamiltonian(self.result, active, onsite_u_ev=3.0)
        # The synthetic solver is explicitly only a workflow stub.
        with self.assertWarns(RuntimeWarning):
            mock = SimulatedADAPTVQESolver()
        q1 = mock.solve(ham)
        feedback = apply_quantum_feedback(self.result, q1, policy_name="occupation", prev_input=self.pw_in)
        self.assertEqual(feedback.metadata["scientific_status"], "experimental_heuristic")

        criteria = ConvergenceCriteria(
            energy_tolerance_ev=1e-4,
            rdm_tolerance=1e-3,
            gradient_tolerance=1e-3,
            max_outer_iterations=5,
            require_gradient=False,  # mock has no scientifically meaningful residual ADAPT gradient
        )
        ledger = OuterLoopLedger(criteria)
        controller = OuterLoopController(ledger=ledger, feedback_policy="occupation_feedback")
        self.assertEqual(controller.step(self.result, q1, prev_input=self.pw_in).decision_type, "NEXT_RUN")
        q2 = mock.solve(ham)
        self.assertEqual(controller.step(self.result, q2, prev_input=self.pw_in).decision_type, "STOP")


if __name__ == "__main__":
    unittest.main()
