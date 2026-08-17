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


if __name__ == "__main__":
    unittest.main()
