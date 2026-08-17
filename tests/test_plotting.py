"""Tests for plotting and convergence visualization (src/qeanalyzer/plotting)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import QERunResult, build_run_result
from qeanalyzer.plotting import plot_relaxation_convergence, plot_scf_convergence

FIXTURES = Path(__file__).parent / "fixtures"


class TestPlotting(unittest.TestCase):
    """Test figure generation for SCF and relaxation convergence."""

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


if __name__ == "__main__":
    unittest.main()
