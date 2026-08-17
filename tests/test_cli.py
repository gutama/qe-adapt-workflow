import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from qeanalyzer import __version__
from qeanalyzer.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


class TestCLI(unittest.TestCase):
    def test_version_is_defined(self):
        self.assertTrue(__version__)

    def test_empty_cli_prints_help(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main([])
        self.assertEqual(code, 0)
        self.assertIn("qeanalyzer", buffer.getvalue())

    def test_dump_stdout(self):
        buffer = io.StringIO()
        args = [
            "dump",
            str(FIXTURES / "si_scf.in"),
            str(FIXTURES / "si_scf.out"),
            str(FIXTURES / "si_scf.xml"),
            "--run-id",
            "cli-test-001",
        ]
        with redirect_stdout(buffer):
            code = main(args)

        self.assertEqual(code, 0)
        output = buffer.getvalue()
        data = json.loads(output)
        self.assertEqual(data["run_id"], "cli-test-001")
        self.assertEqual(data["calculation"], "scf")
        self.assertTrue(data["status"]["scf_converged"])
        self.assertAlmostEqual(data["electronic"]["total_energy_ry"], -15.85434567)

    def test_dump_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "out.json"
            args = [
                "dump",
                str(FIXTURES / "al_scf_metal.out"),
                "-o",
                str(out_file),
                "--run-id",
                "al-cli-001",
            ]
            code = main(args)
            self.assertEqual(code, 0)
            self.assertTrue(out_file.exists())

            data = json.loads(out_file.read_text())
            self.assertEqual(data["run_id"], "al-cli-001")
            self.assertAlmostEqual(data["electronic"]["fermi_energy_ev"], 7.9421)

    def test_dump_directory(self):
        buffer = io.StringIO()
        args = ["dump", str(FIXTURES)]
        with redirect_stdout(buffer):
            code = main(args)
        self.assertEqual(code, 0)
        data = json.loads(buffer.getvalue())
        self.assertIn("calculation", data)

    def test_report_stdout(self):
        buffer = io.StringIO()
        args = [
            "report",
            str(FIXTURES / "si_scf.in"),
            str(FIXTURES / "si_scf.out"),
            str(FIXTURES / "si_scf.xml"),
        ]
        with redirect_stdout(buffer):
            code = main(args)
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("Quantum ESPRESSO Run Analysis", output)
        self.assertIn("Electronic Structure", output)
        self.assertIn("Diagnostics", output)

    def test_report_markdown(self):
        buffer = io.StringIO()
        args = [
            "report",
            str(FIXTURES / "si_scf.out"),
            "--markdown",
        ]
        with redirect_stdout(buffer):
            code = main(args)
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("# Quantum ESPRESSO Run Analysis", output)
        self.assertIn("## Electronic Structure", output)

    def test_report_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "report.md"
            args = [
                "report",
                str(FIXTURES / "si_scf.out"),
                "-o",
                str(out_file),
                "--markdown",
            ]
            code = main(args)
    def test_plot_scf_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "scf.png"
            args = [
                "plot",
                str(FIXTURES / "si_scf.out"),
                "-o",
                str(out_png),
                "--what",
                "scf",
            ]
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(args)
            self.assertEqual(code, 0)
            self.assertTrue(out_png.exists())
            self.assertIn("Saved scf convergence plot", buffer.getvalue())

    def test_plot_relax_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "relax.png"
            args = [
                "plot",
                str(FIXTURES / "si_relax.out"),
                "-o",
                str(out_png),
            ]
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(args)
            self.assertEqual(code, 0)
            self.assertTrue(out_png.exists())
            self.assertIn("Saved relax convergence plot", buffer.getvalue())

    def test_plot_history_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_file = Path(tmpdir) / "workflow.json"
            out_png = Path(tmpdir) / "history.png"

            # Create workflow step to have a populated ledger
            main([
                "next",
                str(FIXTURES / "si_scf.in"),
                str(FIXTURES / "si_scf.out"),
                "--ledger",
                str(ledger_file),
                "-o",
                str(Path(tmpdir) / "002_nscf"),
            ])

            # 1. qeanalyzer plot workflow.json -o history.png
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(["plot", str(ledger_file), "-o", str(out_png)])
            self.assertEqual(code, 0)
            self.assertTrue(out_png.exists())
            self.assertIn("Saved workflow history plot", buffer.getvalue())

            # 2. qeanalyzer plot workflow.json --what history -o history2.png
            out_png2 = Path(tmpdir) / "history2.png"
            buffer2 = io.StringIO()
            with redirect_stdout(buffer2):
                code2 = main(["plot", str(ledger_file), "--what", "history", "-o", str(out_png2)])
            self.assertEqual(code2, 0)
            self.assertTrue(out_png2.exists())

            # 3. qeanalyzer plot-history workflow.json -o history3.png
            out_png3 = Path(tmpdir) / "history3.png"
            buffer3 = io.StringIO()
            with redirect_stdout(buffer3):
                code3 = main(["plot-history", str(ledger_file), "-o", str(out_png3)])
            self.assertEqual(code3, 0)
            self.assertTrue(out_png3.exists())
            self.assertIn("Saved workflow history plot", buffer3.getvalue())

    def test_history_with_plot_flag_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_file = Path(tmpdir) / "workflow.json"
            out_png = Path(tmpdir) / "history.png"

            main([
                "next",
                str(FIXTURES / "si_scf.in"),
                str(FIXTURES / "si_scf.out"),
                "--ledger",
                str(ledger_file),
                "-o",
                str(Path(tmpdir) / "002_nscf"),
            ])

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(["history", str(ledger_file), "--plot", str(out_png)])
            self.assertEqual(code, 0)
            self.assertTrue(out_png.exists())
            self.assertIn("Saved workflow history plot", buffer.getvalue())

    def test_next_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "002_nscf"
            ledger_file = Path(tmpdir) / "workflow.json"
            args = [
                "next",
                str(FIXTURES / "si_scf.in"),
                str(FIXTURES / "si_scf.out"),
                str(FIXTURES / "si_scf.xml"),
                "-o",
                str(out_dir),
                "--ledger",
                str(ledger_file),
            ]
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(args)

            self.assertEqual(code, 0)
            output = buffer.getvalue()
            self.assertIn("Decision : NEXT_RUN", output)
            self.assertIn("Target   : nscf", output)
            self.assertTrue((out_dir / "pw.in").exists())
            self.assertTrue(ledger_file.exists())

    def test_history_and_validate_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_file = Path(tmpdir) / "workflow.json"
            # First create a step
            main([
                "next",
                str(FIXTURES / "si_scf.in"),
                str(FIXTURES / "si_scf.out"),
                "--ledger",
                str(ledger_file),
                "-o",
                str(Path(tmpdir) / "002_nscf"),
            ])

            # History test
            hist_buf = io.StringIO()
            with redirect_stdout(hist_buf):
                hist_code = main(["history", str(ledger_file)])
            self.assertEqual(hist_code, 0)
            self.assertIn("Total Runs", hist_buf.getvalue())

            # Validate test
            val_buf = io.StringIO()
            with redirect_stdout(val_buf):
                val_code = main(["validate", str(ledger_file)])
            self.assertEqual(val_code, 0)
            self.assertIn("is VALID", val_buf.getvalue())

    def test_export_fcidump_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_fci = Path(tmpdir) / "model.fcidump"
            args = [
                "export-fcidump",
                str(FIXTURES / "si_scf.xml"),
                "-o",
                str(out_fci),
                "--active-method",
                "band_index",
                "--band-start",
                "1",
                "--band-end",
                "2",
                "-u",
                "2.0",
            ]
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(args)

            self.assertEqual(code, 0)
            self.assertTrue(out_fci.exists())
            self.assertIn("Exported FCIDUMP", buffer.getvalue())
            content = out_fci.read_text()
            self.assertIn("&FCI", content)
            self.assertIn("NORB=2", content)


if __name__ == "__main__":
    unittest.main()
