"""Tests for the QE pw.x text output parser."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io import (
    ForceAtom,
    IonicStep,
    PWOutput,
    SCFIteration,
    StressTensor,
    parse_pw_output,
    read_pw_output,
)
from qeanalyzer.io.pw_output import _parse_time

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Time parsing helper tests
# ---------------------------------------------------------------------------

class TestTimeParsing(unittest.TestCase):
    """Timing string conversion tests."""

    def test_seconds_only(self):
        self.assertAlmostEqual(_parse_time("0.42s"), 0.42)
        self.assertAlmostEqual(_parse_time("12.5s"), 12.5)

    def test_minutes_and_seconds(self):
        self.assertAlmostEqual(_parse_time("1m 2.34s"), 62.34)
        self.assertAlmostEqual(_parse_time("2m12.50s"), 132.5)

    def test_hours_minutes(self):
        self.assertAlmostEqual(_parse_time("1h 2m"), 3720.0)
        self.assertAlmostEqual(_parse_time("2h30m15s"), 9015.0)

    def test_days_hours(self):
        self.assertAlmostEqual(_parse_time("1d 2h"), 93600.0)

    def test_empty_and_invalid(self):
        self.assertIsNone(_parse_time(""))
        self.assertIsNone(_parse_time("   "))
        self.assertIsNone(_parse_time("not a time"))


# ---------------------------------------------------------------------------
# Fixture tests: Silicon SCF (Insulator)
# ---------------------------------------------------------------------------

class TestParseSiliconSCF(unittest.TestCase):
    """Test parsing of a standard converged silicon SCF calculation."""

    @classmethod
    def setUpClass(cls):
        cls.out = read_pw_output(FIXTURES / "si_scf.out")

    def test_header(self):
        self.assertEqual(self.out.qe_version, "7.2")
        self.assertEqual(self.out.n_mpi_processes, 4)
        self.assertEqual(self.out.n_threads, 1)

    def test_system_info(self):
        self.assertAlmostEqual(self.out.alat_bohr, 10.2)
        self.assertAlmostEqual(self.out.cell_volume_bohr3, 265.302)
        self.assertEqual(self.out.nat, 2)
        self.assertEqual(self.out.ntyp, 1)
        self.assertAlmostEqual(self.out.n_electrons, 8.0)
        self.assertEqual(self.out.n_bands, 4)
        self.assertAlmostEqual(self.out.ecutwfc_ry, 30.0)
        self.assertAlmostEqual(self.out.ecutrho_ry, 240.0)
        self.assertEqual(self.out.n_kpoints, 10)
        self.assertAlmostEqual(self.out.mixing_beta, 0.7)

    def test_scf_iterations(self):
        self.assertEqual(len(self.out.scf_iterations), 8)
        self.assertEqual(self.out.n_scf_iterations, 8)
        self.assertTrue(self.out.scf_converged)

        first = self.out.scf_iterations[0]
        self.assertEqual(first.iteration, 1)
        self.assertAlmostEqual(first.total_energy_ry, -15.82345678)
        self.assertAlmostEqual(first.accuracy_ry, 0.12345678)
        self.assertEqual(first.diag_method, "Davidson")
        self.assertAlmostEqual(first.ethr, 1.0e-2)
        self.assertAlmostEqual(first.avg_diag_iterations, 2.0)

        last = self.out.scf_iterations[-1]
        self.assertEqual(last.iteration, 8)
        self.assertAlmostEqual(last.total_energy_ry, -15.85434567)
        self.assertAlmostEqual(last.accuracy_ry, 1.0e-8)

    def test_energy_breakdown(self):
        self.assertAlmostEqual(self.out.total_energy_ry, -15.85434567)
        self.assertAlmostEqual(self.out.one_electron_ry, 4.83456789)
        self.assertAlmostEqual(self.out.hartree_ry, 1.12345678)
        self.assertAlmostEqual(self.out.xc_ry, -4.84567890)
        self.assertAlmostEqual(self.out.ewald_ry, -16.96668944)
        self.assertIsNone(self.out.smearing_ry)

    def test_electronic_band_edges(self):
        self.assertAlmostEqual(self.out.highest_occupied_ev, 6.2140)
        self.assertAlmostEqual(self.out.lowest_unoccupied_ev, 7.3940)
        self.assertIsNone(self.out.fermi_energy_ev)

    def test_forces(self):
        self.assertIsNotNone(self.out.forces)
        self.assertEqual(len(self.out.forces), 2)
        self.assertEqual(self.out.forces[0].symbol, "Si")
        self.assertEqual(self.out.forces[0].atom_index, 1)
        self.assertEqual(self.out.forces[0].force, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(self.out.total_force, 0.0)
        self.assertAlmostEqual(self.out.scf_correction_force, 0.000001)

    def test_timing_and_status(self):
        self.assertAlmostEqual(self.out.cpu_time_seconds, 0.42)
        self.assertAlmostEqual(self.out.wall_time_seconds, 0.54)
        self.assertTrue(self.out.job_done)
        self.assertEqual(self.out.exit_status, "converged")


# ---------------------------------------------------------------------------
# Fixture tests: Aluminum SCF (Metal with Smearing)
# ---------------------------------------------------------------------------

class TestParseAluminumMetalSCF(unittest.TestCase):
    """Test parsing of an aluminum metal calculation with smearing."""

    @classmethod
    def setUpClass(cls):
        cls.out = read_pw_output(FIXTURES / "al_scf_metal.out")

    def test_system_info(self):
        self.assertEqual(self.out.nat, 1)
        self.assertEqual(self.out.ntyp, 1)
        self.assertAlmostEqual(self.out.n_electrons, 3.0)
        self.assertEqual(self.out.n_bands, 6)
        self.assertEqual(self.out.n_kpoints, 60)

    def test_fermi_and_smearing(self):
        self.assertAlmostEqual(self.out.fermi_energy_ev, 7.9421)
        self.assertIsNone(self.out.highest_occupied_ev)
        self.assertAlmostEqual(self.out.smearing_ry, -0.00012345)

    def test_scf_and_energy(self):
        self.assertEqual(len(self.out.scf_iterations), 6)
        self.assertTrue(self.out.scf_converged)
        self.assertAlmostEqual(self.out.total_energy_ry, -4.18693542)

    def test_timing_and_status(self):
        self.assertAlmostEqual(self.out.cpu_time_seconds, 0.31)
        self.assertAlmostEqual(self.out.wall_time_seconds, 0.38)
        self.assertTrue(self.out.job_done)
        self.assertEqual(self.out.exit_status, "converged")


# ---------------------------------------------------------------------------
# Fixture tests: Silicon Ionic Relax
# ---------------------------------------------------------------------------

class TestParseSiliconRelax(unittest.TestCase):
    """Test parsing of an ionic relaxation calculation."""

    @classmethod
    def setUpClass(cls):
        cls.out = read_pw_output(FIXTURES / "si_relax.out")

    def test_ionic_steps(self):
        self.assertTrue(self.out.bfgs_converged)
        self.assertEqual(self.out.n_bfgs_steps, 3)
        self.assertIsNotNone(self.out.ionic_steps)
        self.assertEqual(len(self.out.ionic_steps), 3)

    def test_forces_decrease_across_steps(self):
        steps = self.out.ionic_steps
        f1 = steps[0].total_force
        f2 = steps[1].total_force
        f3 = steps[2].total_force
        self.assertGreater(f1, f2)
        self.assertGreater(f2, f3)

    def test_final_state(self):
        self.assertTrue(self.out.scf_converged)
        self.assertAlmostEqual(self.out.total_energy_ry, -15.85434567)
        self.assertEqual(self.out.exit_status, "converged")


# ---------------------------------------------------------------------------
# Fixture tests: Silicon VC-Relax (Cell + Ionic)
# ---------------------------------------------------------------------------

class TestParseSiliconVCRelax(unittest.TestCase):
    """Test parsing of a variable-cell relaxation calculation."""

    @classmethod
    def setUpClass(cls):
        cls.out = read_pw_output(FIXTURES / "si_vc_relax.out")

    def test_bfgs_converged(self):
        self.assertTrue(self.out.bfgs_converged)
        self.assertEqual(self.out.n_bfgs_steps, 2)
        self.assertEqual(len(self.out.ionic_steps), 2)

    def test_stress_tensor(self):
        self.assertIsNotNone(self.out.stress)
        self.assertAlmostEqual(self.out.stress.pressure_kbar, -0.02)
        tensor = self.out.stress.tensor_kbar
        self.assertEqual(len(tensor), 3)
        self.assertEqual(len(tensor[0]), 3)
        self.assertAlmostEqual(tensor[0][0], 0.02)
        self.assertAlmostEqual(tensor[1][1], 0.02)
        self.assertAlmostEqual(tensor[2][2], -0.08)

    def test_ionic_step_stress(self):
        first_step_stress = self.out.ionic_steps[0].stress
        self.assertIsNotNone(first_step_stress)
        self.assertAlmostEqual(first_step_stress.pressure_kbar, 15.20)


# ---------------------------------------------------------------------------
# Fixture tests: Failed SCF
# ---------------------------------------------------------------------------

class TestParseFailedSCF(unittest.TestCase):
    """Test parsing of an unconverged SCF calculation."""

    @classmethod
    def setUpClass(cls):
        cls.out = read_pw_output(FIXTURES / "failed_scf.out")

    def test_not_converged(self):
        self.assertFalse(self.out.scf_converged)
        self.assertEqual(len(self.out.scf_iterations), 10)
        self.assertTrue(self.out.job_done)
        self.assertEqual(self.out.exit_status, "not_converged")


# ---------------------------------------------------------------------------
# Fixture tests: Interrupted Run
# ---------------------------------------------------------------------------

class TestParseInterruptedRun(unittest.TestCase):
    """Test parsing of a truncated/interrupted calculation."""

    @classmethod
    def setUpClass(cls):
        cls.out = read_pw_output(FIXTURES / "interrupted.out")

    def test_interrupted_status(self):
        self.assertFalse(self.out.job_done)
        self.assertFalse(self.out.scf_converged)
        self.assertEqual(self.out.exit_status, "interrupted")

    def test_partial_data_extracted(self):
        self.assertEqual(self.out.qe_version, "7.2")
        self.assertEqual(self.out.nat, 2)
        self.assertEqual(len(self.out.scf_iterations), 3)


# ---------------------------------------------------------------------------
# Synthetic & Edge Case Tests
# ---------------------------------------------------------------------------

class TestSyntheticOutputs(unittest.TestCase):
    """Edge cases with synthetic text snippets."""

    def test_empty_string(self):
        out = parse_pw_output("")
        self.assertFalse(out.job_done)
        self.assertEqual(out.exit_status, "interrupted")
        self.assertEqual(len(out.scf_iterations), 0)

    def test_magnetization(self):
        text = (
            "     total magnetization       =     1.00 Bohr mag/cell\n"
            "     absolute magnetization    =     1.25 Bohr mag/cell\n"
        )
        out = parse_pw_output(text)
        self.assertAlmostEqual(out.total_magnetization, 1.0)
        self.assertAlmostEqual(out.absolute_magnetization, 1.25)

    def test_warnings_and_errors(self):
        text = (
            "     Warning: symmetry not detected\n"
            "     % Warning: ecutwfc might be too small\n"
            "     Error in routine cdiaghg (1):\n"
        )
        out = parse_pw_output(text)
        self.assertEqual(len(out.warnings), 2)
        self.assertEqual(len(out.errors), 1)
        self.assertEqual(out.exit_status, "error")

    def test_cg_diagonalization(self):
        text = (
            "     iteration #  1     ecut=    30.00 Ry     beta= 0.70\n"
            "     CG style diagonalization\n"
            "     ethr =  1.00E-02,  avg # of iterations =  2.0\n"
            "     total energy              =     -10.00000000 Ry\n"
            "     estimated scf accuracy    <       0.01000000 Ry\n"
        )
        out = parse_pw_output(text)
        self.assertEqual(len(out.scf_iterations), 1)
        self.assertEqual(out.scf_iterations[0].diag_method, "CG")

class TestPerIonicStepConvergence(unittest.TestCase):
    """Each archived ionic step records its own electronic convergence.

    Hardcoding converged=True gave every intermediate step a clean bill of
    health, so a relaxation that limped through unconverged electronic steps
    looked identical to a healthy one.
    """

    @classmethod
    def setUpClass(cls):
        cls.out = read_pw_output(FIXTURES / "relax_mixed_convergence.out")

    def test_both_ionic_steps_are_recorded(self):
        self.assertEqual(len(self.out.ionic_steps), 2)

    def test_failed_electronic_step_is_marked_unconverged(self):
        self.assertIs(self.out.ionic_steps[0].converged, False)

    def test_successful_electronic_step_is_marked_converged(self):
        self.assertIs(self.out.ionic_steps[1].converged, True)

    def test_steps_are_distinguishable(self):
        """The whole point: a caller can tell which geometry step struggled."""
        flags = [s.converged for s in self.out.ionic_steps]
        self.assertNotEqual(len(set(flags)), 1)

    def test_healthy_relaxation_has_all_steps_converged(self):
        healthy = read_pw_output(FIXTURES / "relax_unconverged.out")
        self.assertTrue(all(s.converged for s in healthy.ionic_steps))


if __name__ == "__main__":
    unittest.main()
