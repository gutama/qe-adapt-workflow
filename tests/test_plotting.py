"""Tests for plotting and convergence visualization (src/qeanalyzer/plotting)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import QERunResult, build_run_result
from qeanalyzer.plotting import (
    plot_relaxation_convergence,
    plot_scf_convergence,
    plot_workflow_history,
)
from qeanalyzer.workflow import WorkflowLedger, WorkflowRunEntry

FIXTURES = Path(__file__).parent / "fixtures"


class TestPlotting(unittest.TestCase):
    """Test figure generation for SCF, relaxation, and workflow history."""

    def test_plot_scf_convergence(self):
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        result = build_run_result(pw_out=pw_out, run_id="si-scf")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "scf_conv.png"
            fig = plot_scf_convergence(result, output_path=out_png)
            self.assertIsNotNone(fig)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 1000)

    def test_plot_relaxation_convergence(self):
        pw_out = read_pw_output(FIXTURES / "si_relax.out")
        result = build_run_result(pw_out=pw_out, run_id="si-relax")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "relax_conv.png"
            fig = plot_relaxation_convergence(result, output_path=out_png)
            self.assertIsNotNone(fig)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 1000)

    def test_plot_vc_relaxation_convergence(self):
        pw_out = read_pw_output(FIXTURES / "si_vc_relax.out")
        result = build_run_result(pw_out=pw_out, run_id="si-vc-relax")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "vc_relax_conv.png"
            fig = plot_relaxation_convergence(result, output_path=out_png)
            self.assertIsNotNone(fig)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 1000)

    def test_plot_scf_empty_raises(self):
        empty_result = QERunResult()
        with self.assertRaises(ValueError):
            plot_scf_convergence(empty_result)

    def test_plot_relax_empty_raises(self):
        empty_result = QERunResult()
        with self.assertRaises(ValueError):
            plot_relaxation_convergence(empty_result)

    def test_plot_workflow_history_from_ledger_obj(self):
        ledger = WorkflowLedger(workflow_id="silicon_chain")
        ledger.add_run(WorkflowRunEntry(
            run_id="qe-0001",
            calculation="vc-relax",
            status="converged",
            energy_ry=-15.8420,
        ))
        ledger.add_run(WorkflowRunEntry(
            run_id="qe-0002",
            calculation="scf",
            parent_run="qe-0001",
            status="converged",
            energy_ry=-15.8427,
            policy_applied="relax_to_scf",
        ))
        ledger.add_run(WorkflowRunEntry(
            run_id="qe-0003",
            calculation="nscf",
            parent_run="qe-0002",
            status="converged",
            energy_ry=-15.8427,
            policy_applied="scf_to_nscf",
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "history.png"
            fig = plot_workflow_history(ledger, output_path=out_png)
            self.assertIsNotNone(fig)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 1000)

    def test_plot_workflow_history_from_json_path(self):
        ledger = WorkflowLedger(workflow_id="silicon_test")
        ledger.add_run(WorkflowRunEntry(run_id="r1", calculation="scf", status="converged", energy_ry=-10.0))
        ledger.add_run(WorkflowRunEntry(run_id="r2", calculation="nscf", parent_run="r1", status="converged", energy_ry=-10.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            l_json = Path(tmpdir) / "workflow.json"
            ledger.save_json(l_json)
            out_png = Path(tmpdir) / "history.png"
            fig = plot_workflow_history(l_json, output_path=out_png)
            self.assertIsNotNone(fig)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 1000)

    def test_plot_workflow_history_mixed_statuses(self):
        ledger = WorkflowLedger(workflow_id="mixed_workflow")
        ledger.add_run(WorkflowRunEntry(run_id="r1", calculation="scf", status="converged", energy_ry=-10.5))
        ledger.add_run(WorkflowRunEntry(run_id="r2", calculation="scf", parent_run="r1", status="not_converged", energy_ry=-10.4))
        ledger.add_run(WorkflowRunEntry(run_id="r3", calculation="scf", parent_run="r2", status="interrupted", energy_ry=None))
        ledger.add_run(WorkflowRunEntry(run_id="r4", calculation="scf", parent_run="r3", status="planned", energy_ry=None))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "mixed_history.png"
            fig = plot_workflow_history(ledger, output_path=out_png)
            self.assertIsNotNone(fig)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 1000)

    def test_plot_workflow_history_no_energies(self):
        ledger = WorkflowLedger(workflow_id="planned_workflow")
        ledger.add_run(WorkflowRunEntry(run_id="p1", calculation="vc-relax", status="planned"))
        ledger.add_run(WorkflowRunEntry(run_id="p2", calculation="scf", parent_run="p1", status="planned", policy_applied="relax_to_scf"))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "planned_history.png"
            fig = plot_workflow_history(ledger, output_path=out_png)
            self.assertIsNotNone(fig)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 1000)

    def test_plot_workflow_history_empty_raises(self):
        empty_ledger = WorkflowLedger(workflow_id="empty")
        with self.assertRaises(ValueError):
            plot_workflow_history(empty_ledger)

    def test_plot_workflow_history_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            plot_workflow_history(12345)  # type: ignore


if __name__ == "__main__":
    unittest.main()
