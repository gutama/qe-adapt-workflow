"""Tests for workflow ledger, DAG validation, and lineage tracking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.workflow import (
    WorkflowLedger,
    WorkflowRunEntry,
    plan_next_calculation,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestWorkflowLedger(unittest.TestCase):
    """Test ledger recording, DAG traversal, validation, and serialization."""

    def test_add_and_lookup_runs(self):
        ledger = WorkflowLedger(workflow_id="si-chain-01")
        entry1 = WorkflowRunEntry(
            run_id="run-001",
            calculation="vc-relax",
            status="converged",
            energy_ry=-15.8543,
        )
        entry2 = WorkflowRunEntry(
            run_id="run-002",
            calculation="scf",
            parent_run="run-001",
            status="converged",
            energy_ry=-15.85434567,
        )
        ledger.add_run(entry1)
        ledger.add_run(entry2)

        self.assertEqual(len(ledger.runs), 2)
        self.assertEqual(ledger.get_run("run-001").calculation, "vc-relax")
        self.assertEqual(ledger.get_latest_run().run_id, "run-002")

    def test_dag_lineage_traversal(self):
        ledger = WorkflowLedger(workflow_id="lineage-test")
        ledger.add_run(WorkflowRunEntry(run_id="r1", calculation="relax"))
        ledger.add_run(WorkflowRunEntry(run_id="r2", calculation="scf", parent_run="r1"))
        ledger.add_run(WorkflowRunEntry(run_id="r3", calculation="nscf", parent_run="r2"))

        lineage = ledger.get_lineage("r3")
        self.assertEqual(len(lineage), 3)
        self.assertEqual([r.run_id for r in lineage], ["r1", "r2", "r3"])

    def test_validation_clean_dag(self):
        ledger = WorkflowLedger(workflow_id="valid-dag")
        ledger.add_run(WorkflowRunEntry(run_id="r1", calculation="scf"))
        ledger.add_run(WorkflowRunEntry(run_id="r2", calculation="nscf", parent_run="r1"))
        errors = ledger.validate()
        self.assertEqual(errors, [])

    def test_validation_duplicate_id(self):
        ledger = WorkflowLedger(workflow_id="dup-dag")
        ledger.runs.append(WorkflowRunEntry(run_id="r1", calculation="scf"))
        ledger.runs.append(WorkflowRunEntry(run_id="r1", calculation="scf"))
        errors = ledger.validate()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Duplicate" in e for e in errors))

    def test_validation_unknown_parent(self):
        ledger = WorkflowLedger(workflow_id="bad-parent")
        ledger.add_run(WorkflowRunEntry(run_id="r1", calculation="scf", parent_run="non_existent"))
        errors = ledger.validate()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("parent_run" in e for e in errors))

    def test_validation_cyclic_dependency(self):
        ledger = WorkflowLedger(workflow_id="cycle-dag")
        ledger.runs.append(WorkflowRunEntry(run_id="r1", calculation="scf", parent_run="r2"))
        ledger.runs.append(WorkflowRunEntry(run_id="r2", calculation="scf", parent_run="r1"))
        errors = ledger.validate()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Cyclic" in e for e in errors))

    def test_json_save_load_roundtrip(self):
        ledger = WorkflowLedger(workflow_id="save-load-test", description="Test description")
        ledger.add_run(WorkflowRunEntry(run_id="r1", calculation="scf", status="converged"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "workflow.json"
            ledger.save_json(path)
            self.assertTrue(path.exists())

            loaded = WorkflowLedger.load_json(path)
            self.assertEqual(loaded.workflow_id, "save-load-test")
            self.assertEqual(loaded.description, "Test description")
            self.assertEqual(len(loaded.runs), 1)
            self.assertEqual(loaded.runs[0].run_id, "r1")

    def test_plan_next_calculation_with_staging_and_ledger(self):
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")
        result = build_run_result(pw_in=pw_in, pw_out=pw_out, qe_xml=qe_xml, run_id="si-scf-001")

        ledger = WorkflowLedger(workflow_id="orchestrate-test")
        ledger.add_run(WorkflowRunEntry(run_id="si-scf-001", calculation="scf", status="converged"))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "runs" / "002_nscf"
            decision = plan_next_calculation(
                result=result,
                prev_input=pw_in,
                output_dir=out_dir,
                ledger=ledger,
                next_run_id="si-nscf-002",
            )

            self.assertEqual(decision.decision_type, "NEXT_RUN")
            self.assertEqual(decision.target_calculation, "nscf")
            self.assertTrue((out_dir / "pw.in").exists())
            self.assertEqual(len(ledger.runs), 2)
            self.assertEqual(ledger.get_latest_run().run_id, "si-nscf-002")


if __name__ == "__main__":
    unittest.main()
