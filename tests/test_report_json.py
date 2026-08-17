"""Tests for JSON reporting and serialization (src/qeanalyzer/report/json.py)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.report import (
    dump_result_json,
    load_result_json,
    save_result_json,
    validate_result_dict,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestJSONReporting(unittest.TestCase):
    """Test JSON serialization, file operations, determinism, and validation."""

    @classmethod
    def setUpClass(cls):
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")
        cls.result = build_run_result(
            pw_in=pw_in,
            pw_out=pw_out,
            qe_xml=qe_xml,
            run_id="test-si-001",
            parent_run="root",
        )

    def test_dump_result_json_deterministic(self):
        json_str1 = dump_result_json(self.result, indent=2)
        json_str2 = dump_result_json(self.result, indent=2)
        self.assertEqual(json_str1, json_str2)

        # Keys should be sorted in output
        parsed = json.loads(json_str1)
        self.assertEqual(parsed["run_id"], "test-si-001")
        self.assertEqual(parsed["calculation"], "scf")

    def test_save_and_load_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "result.json"
            save_result_json(self.result, file_path)
            self.assertTrue(file_path.exists())

            loaded = load_result_json(file_path)
            self.assertEqual(loaded.run_id, "test-si-001")
            self.assertEqual(loaded.calculation, "scf")
            self.assertTrue(loaded.status.scf_converged)
            self.assertAlmostEqual(
                loaded.electronic.total_energy_ry,
                self.result.electronic.total_energy_ry,
            )

    def test_validate_result_dict_valid(self):
        d = self.result.to_dict()
        errors = validate_result_dict(d)
        self.assertEqual(errors, [])

    def test_validate_result_dict_missing_fields(self):
        invalid_d = {
            "schema_version": "1.0",
            # missing calculation, status, etc.
        }
        errors = validate_result_dict(invalid_d)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("calculation" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
