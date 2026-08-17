"""End-to-end multi-stage integration tests (Stages A through E)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.plotting import plot_relaxation_convergence, plot_scf_convergence, plot_workflow_history
from qeanalyzer.quantum import (
    ADAPTVQESolver,
    ExactDiagonalizationSolver,
    apply_quantum_feedback,
    build_active_space_hamiltonian,
    read_fcidump,
    select_active_space,
    write_fcidump,
)
from qeanalyzer.report import dump_result_json, generate_text_report
from qeanalyzer.runner import LocalRunner, RunSpec
from qeanalyzer.workflow import (
    ConvergenceCriteria,
    OuterLoopController,
    OuterLoopLedger,
    WorkflowLedger,
    plan_next_calculation,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestEndToEndPipeline(unittest.TestCase):
    """Verify entire pipeline from DFT parsing to ADAPT-VQE and outer-loop convergence."""

    def test_stage_a_dft_parsing_reporting_and_nscf_transition(self):
        """Stage A: QE calculation -> Parser -> Report -> Plot -> NSCF generation."""
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")

        result = build_run_result(pw_in=pw_in, pw_out=pw_out, qe_xml=qe_xml, run_id="si_scf_001")
        self.assertTrue(result.status.scf_converged)

        # 1. Report generation
        rep_md = generate_text_report(result, markdown=True)
        self.assertIn("Quantum ESPRESSO Run Analysis", rep_md)
        self.assertIn("CONVERGED", rep_md)

        # 2. JSON dump
        json_data = dump_result_json(result)
        self.assertIn('"schema_version": "1.0"', json_data)

        # 3. Plotting
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            p_path = tmp_path / "scf_conv.png"
            plot_scf_convergence(result, output_path=p_path)
            self.assertTrue(p_path.exists())

            # 4. Next-run decision and ledger
            ledger = WorkflowLedger(workflow_id="wf_silicon_01")
            decision = plan_next_calculation(
                result=result,
                prev_input=pw_in,
                output_dir=tmp_path / "002_nscf",
                ledger=ledger,
            )
            self.assertEqual(decision.decision_type, "NEXT_RUN")
            self.assertEqual(decision.policy_name, "scf_to_nscf")
            self.assertTrue((tmp_path / "002_nscf" / "pw.in").exists())
            self.assertEqual(len(ledger.runs), 1)

    def test_stage_b_active_space_and_exact_diagonalization(self):
        """Stage B: Electronic state -> Active Space -> Hamiltonian -> Exact Diagonalization."""
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")
        result = build_run_result(pw_in=pw_in, pw_out=pw_out, qe_xml=qe_xml, run_id="si_scf")

        # Select 2-band active space
        asp = select_active_space(result, method="band_index", band_start=1, band_end=2)
        self.assertEqual(asp.n_active_orbitals, 2)

        # Build Hamiltonian with onsite U and intersite V
        ham = build_active_space_hamiltonian(result, active_space=asp, onsite_u_ev=2.0, intersite_v_ev=0.5)
        self.assertTrue(ham.is_hermitian())

        # Exact FCI solver
        ed_solver = ExactDiagonalizationSolver()
        q_res = ed_solver.solve(ham, active_space=asp)
        self.assertTrue(q_res.converged)
        self.assertEqual(q_res.n_orbitals, 2)
        self.assertEqual(len(q_res.natural_occupations), 2)

    def test_stage_c_adapt_vqe_and_fcidump_interchange(self):
        """Stage C: Hamiltonian -> ADAPT-VQE -> FCIDUMP interchange roundtrip."""
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")
        result = build_run_result(pw_in=pw_in, pw_out=pw_out, qe_xml=qe_xml, run_id="si_scf")

        asp = select_active_space(result, method="band_index", band_start=1, band_end=2)
        ham = build_active_space_hamiltonian(result, active_space=asp, onsite_u_ev=1.5)

        # FCIDUMP roundtrip
        with tempfile.TemporaryDirectory() as tmpdir:
            fci_file = Path(tmpdir) / "si.fcidump"
            write_fcidump(ham, path=fci_file)
            self.assertTrue(fci_file.exists())
            reloaded_ham = read_fcidump(fci_file)
            self.assertEqual(reloaded_ham.n_orbitals, 2)

            # ADAPT-VQE solver
            adapt_solver = ADAPTVQESolver(gradient_threshold=1e-3, max_adapt_iterations=10)
            vqe_res = adapt_solver.solve(reloaded_ham, active_space=asp)
            self.assertTrue(vqe_res.converged)
            self.assertGreater(len(vqe_res.selected_operators), 0)

    def test_stage_d_and_e_quantum_feedback_and_outer_loop_convergence(self):
        """Stage D & E: Quantum feedback -> Outer-loop self-consistent cycle -> Convergence."""
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")
        dft_res = build_run_result(pw_in=pw_in, pw_out=pw_out, qe_xml=qe_xml, run_id="dft_001")

        # 1. Quantum feedback rule
        q_res_fractional = ADAPTVQESolver().solve(
            build_active_space_hamiltonian(
                dft_res,
                active_space=select_active_space(dft_res, method="band_index", band_start=1, band_end=2),
                onsite_u_ev=3.0,
            )
        )
        dec_fb = apply_quantum_feedback(dft_res, q_res_fractional, policy_name="occupation", prev_input=pw_in)
        self.assertEqual(dec_fb.decision_type, "NEXT_RUN")
        self.assertEqual(dec_fb.policy_name, "occupation_feedback")

        # 2. Outer loop controller orchestration
        criteria = ConvergenceCriteria(energy_tolerance_ev=1e-4, rdm_tolerance=1e-3, max_outer_iterations=5)
        outer_ledger = OuterLoopLedger(criteria=criteria)
        controller = OuterLoopController(ledger=outer_ledger, feedback_policy="occupation_feedback")

        # Step 1: Initial step
        dec1 = controller.step(dft_res, q_res_fractional, prev_input=pw_in)
        self.assertEqual(dec1.decision_type, "NEXT_RUN")
        self.assertFalse(outer_ledger.check_convergence().is_converged)

        # Step 2: Converged step with small delta E
        q_res_converged = ADAPTVQESolver().solve(
            build_active_space_hamiltonian(
                dft_res,
                active_space=select_active_space(dft_res, method="band_index", band_start=1, band_end=2),
                onsite_u_ev=3.0,
            )
        )
        # Identical Hamiltonian produces zero delta E -> triggers outer-loop convergence!
        dec2 = controller.step(dft_res, q_res_converged, prev_input=pw_in)
        self.assertEqual(dec2.decision_type, "STOP")
        self.assertIn("Outer Loop Converged", dec2.reason)
        self.assertTrue(outer_ledger.check_convergence().is_converged)


if __name__ == "__main__":
    unittest.main()
