"""Tests for human-readable text and Markdown reports (src/qeanalyzer/report/text.py)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.report import generate_text_report, save_text_report

FIXTURES = Path(__file__).parent / "fixtures"


class TestTextReporting(unittest.TestCase):
    """Test text and Markdown report generation."""

    @classmethod
    def setUpClass(cls):
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")
        cls.result = build_run_result(
            pw_in=pw_in,
            pw_out=pw_out,
            qe_xml=qe_xml,
            run_id="si-001",
            parent_run="root",
        )

    def test_generate_plain_text_report(self):
        report = generate_text_report(self.result, markdown=False)
        self.assertIn("Quantum ESPRESSO Run Analysis", report)
        self.assertIn("Run Metadata", report)
        self.assertIn("Electronic Structure", report)
        self.assertIn("Total energy", report)
        self.assertIn("-15.85434567", report)
        self.assertIn("Band gap", report)
        self.assertIn("Diagnostics", report)
        self.assertIn("✓ [SCF] SCF converged", report)
        self.assertIn("Recommended Next Step", report)

    def test_generate_markdown_report(self):
        report = generate_text_report(self.result, markdown=True)
        self.assertIn("# Quantum ESPRESSO Run Analysis", report)
        self.assertIn("## Electronic Structure", report)
        self.assertIn("- **✓** [SCF] SCF converged", report)

    def test_save_report_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            save_text_report(self.result, path, markdown=True)
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn("# Quantum ESPRESSO Run Analysis", content)


if __name__ == "__main__":
    unittest.main()
