"""Tests for the Quantum ESPRESSO XML data parser (qe_xml.py)."""

from __future__ import annotations

import unittest
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from qeanalyzer.io import (
    QEXMLKPoint,
    QEXMLOutput,
    QEXMLStructure,
    parse_qe_xml,
    read_qe_xml,
)
from qeanalyzer.io.qe_xml import HARTREE_TO_EV, HARTREE_TO_RY

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture tests: Silicon SCF XML
# ---------------------------------------------------------------------------

class TestParseSiliconSCFXML(unittest.TestCase):
    """Test parsing of Silicon SCF data-file-schema.xml fixture."""

    @classmethod
    def setUpClass(cls):
        cls.xml = read_qe_xml(FIXTURES / "si_scf.xml")

    def test_schema_and_creator(self):
        self.assertEqual(self.xml.schema_version, "1.0.0")
        self.assertEqual(self.xml.creator_name, "PWSCF")
        self.assertEqual(self.xml.creator_version, "7.2")
        self.assertEqual(self.xml.job_title, "silicon")
        self.assertEqual(self.xml.nprocs, 4)
        self.assertEqual(self.xml.nthreads, 1)

    def test_input_parameters(self):
        self.assertEqual(self.xml.calculation, "scf")
        self.assertEqual(self.xml.prefix, "silicon")
        self.assertAlmostEqual(self.xml.ecutwfc_hartree, 15.0)
        self.assertAlmostEqual(self.xml.ecutrho_hartree, 120.0)
        self.assertAlmostEqual(self.xml.conv_thr, 1.0e-8)
        self.assertAlmostEqual(self.xml.mixing_beta, 0.7)

    def test_convergence(self):
        self.assertTrue(self.xml.scf_converged)
        self.assertEqual(self.xml.n_scf_steps, 8)
        self.assertAlmostEqual(self.xml.scf_error, 1.0e-9)

    def test_total_energy_and_units(self):
        self.assertAlmostEqual(self.xml.total_energy_hartree, -7.927172835)
        self.assertAlmostEqual(self.xml.total_energy_ry, -15.85434567)
        self.assertAlmostEqual(
            self.xml.total_energy_ev,
            -7.927172835 * HARTREE_TO_EV,
        )
        self.assertAlmostEqual(self.xml.one_electron_hartree, 2.417283945)
        self.assertAlmostEqual(self.xml.hartree_energy_hartree, 0.561728390)
        self.assertAlmostEqual(self.xml.xc_energy_hartree, -2.422839450)
        self.assertAlmostEqual(self.xml.ewald_energy_hartree, -8.483344720)

    def test_band_structure_and_gap(self):
        self.assertAlmostEqual(self.xml.n_electrons, 8.0)
        self.assertEqual(self.xml.n_bands, 4)
        self.assertEqual(self.xml.n_kpoints, 2)
        self.assertFalse(self.xml.lsda)
        self.assertFalse(self.xml.noncolin)

        # Levels in Hartree and eV
        self.assertAlmostEqual(self.xml.highest_occupied_hartree, 0.2283602)
        self.assertAlmostEqual(self.xml.lowest_unoccupied_hartree, 0.2717255)
        self.assertAlmostEqual(self.xml.highest_occupied_ev, 0.2283602 * HARTREE_TO_EV)
        self.assertAlmostEqual(self.xml.lowest_unoccupied_ev, 0.2717255 * HARTREE_TO_EV)

        # Band gap ~ 1.18 eV
        expected_gap = (0.2717255 - 0.2283602) * HARTREE_TO_EV
        self.assertAlmostEqual(self.xml.band_gap_ev, expected_gap)

    def test_kpoints_eigenvalues(self):
        self.assertEqual(len(self.xml.kpoints), 2)

        kp0 = self.xml.kpoints[0]
        self.assertEqual(kp0.point, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(kp0.weight, 0.25)
        self.assertEqual(len(kp0.eigenvalues_hartree), 4)
        self.assertEqual(len(kp0.eigenvalues_ev), 4)
        self.assertAlmostEqual(kp0.eigenvalues_hartree[0], -0.215430)
        self.assertAlmostEqual(kp0.eigenvalues_ev[0], -0.215430 * HARTREE_TO_EV)
        self.assertEqual(kp0.occupations, [2.0, 2.0, 2.0, 2.0])

        kp1 = self.xml.kpoints[1]
        self.assertEqual(kp1.point, (0.25, 0.25, 0.25))
        self.assertAlmostEqual(kp1.weight, 0.75)

    def test_initial_structure(self):
        struct = self.xml.initial_structure
        self.assertIsNotNone(struct)
        self.assertAlmostEqual(struct.alat_bohr, 10.2)
        self.assertEqual(len(struct.positions_bohr), 2)
        self.assertEqual(struct.positions_bohr[0][0], "Si")
        self.assertEqual(struct.positions_bohr[0][1], (0.0, 0.0, 0.0))
        self.assertEqual(struct.positions_bohr[1][1], (2.55, 2.55, 2.55))

    def test_forces_and_stress(self):
        self.assertIsNotNone(self.xml.forces_hartree_bohr)
        self.assertEqual(len(self.xml.forces_hartree_bohr), 2)
        self.assertEqual(self.xml.forces_ry_bohr, [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
        self.assertIsNotNone(self.xml.stress_kbar)


# ---------------------------------------------------------------------------
# Fixture tests: Aluminum Metal XML
# ---------------------------------------------------------------------------

class TestParseAluminumMetalXML(unittest.TestCase):
    """Test parsing of Aluminum Metal XML with smearing and Fermi energy."""

    @classmethod
    def setUpClass(cls):
        cls.xml = read_qe_xml(FIXTURES / "al_scf_metal.xml")

    def test_fermi_energy(self):
        self.assertAlmostEqual(self.xml.fermi_energy_hartree, 0.2918664)
        expected_fermi_ev = 0.2918664 * HARTREE_TO_EV
        self.assertAlmostEqual(self.xml.fermi_energy_ev, expected_fermi_ev)
        self.assertIsNone(self.xml.highest_occupied_hartree)
        self.assertIsNone(self.xml.band_gap_ev)

    def test_demet_smearing_energy(self):
        self.assertAlmostEqual(self.xml.demet_energy_hartree, -0.000061725)

    def test_fractional_occupations(self):
        kp0 = self.xml.kpoints[0]
        self.assertEqual(len(kp0.occupations), 6)
        self.assertAlmostEqual(kp0.occupations[0], 2.0)
        self.assertAlmostEqual(kp0.occupations[1], 0.333333)


# ---------------------------------------------------------------------------
# Fixture tests: Silicon VC-Relax XML
# ---------------------------------------------------------------------------

class TestParseSiliconVCRelaxXML(unittest.TestCase):
    """Test parsing of VC-Relax XML with initial and final structures."""

    @classmethod
    def setUpClass(cls):
        cls.xml = read_qe_xml(FIXTURES / "si_vc_relax.xml")

    def test_opt_convergence(self):
        self.assertTrue(self.xml.opt_converged)
        self.assertEqual(self.xml.n_opt_steps, 2)

    def test_initial_vs_final_structure(self):
        init_s = self.xml.initial_structure
        final_s = self.xml.final_structure

        self.assertIsNotNone(init_s)
        self.assertIsNotNone(final_s)

        # Lattice vector changed in final structure
        self.assertAlmostEqual(init_s.cell_vectors_bohr[0][1], 5.10)
        self.assertAlmostEqual(final_s.cell_vectors_bohr[0][1], 5.105)

    def test_stress_tensor(self):
        self.assertIsNotNone(self.xml.stress_kbar)
        self.assertIsNotNone(self.xml.pressure_kbar)
        # Pressure is scalar float
        self.assertIsInstance(self.xml.pressure_kbar, float)


# ---------------------------------------------------------------------------
# Synthetic & Edge Case Tests
# ---------------------------------------------------------------------------

class TestXMLParserEdgeCases(unittest.TestCase):
    """Edge cases for XML parser."""

    def test_minimal_xml(self):
        minimal = "<qes:espresso xmlns:qes='http://test' SCHEMA_VERSION='1.0.0'/>"
        out = parse_qe_xml(minimal)
        self.assertEqual(out.schema_version, "1.0.0")
        self.assertFalse(out.scf_converged)
        self.assertIsNone(out.total_energy_hartree)
        self.assertIsNone(out.total_energy_ry)

    def test_unsupported_schema_version_warning(self):
        xml_text = "<qes:espresso xmlns:qes='http://www.quantum-espresso.org/ns/qes/qes-1.0' SCHEMA_VERSION='99.0.0'/>"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = parse_qe_xml(xml_text)
            self.assertEqual(out.schema_version, "99.0.0")
            self.assertTrue(any("schema version" in str(item.message).lower() for item in w))

    def test_invalid_xml_raises(self):
        with self.assertRaises(ET.ParseError):
            parse_qe_xml("not valid xml <<<")

_XML_HEAD = (
    "<qes:espresso xmlns:qes='http://www.quantum-espresso.org/ns/qes/qes-1.0' "
    "SCHEMA_VERSION='23.02.00'>"
)


def _xml_with_convergence(scf_flag: str | None, opt_flag: str | None = None) -> str:
    """Build a minimal XML whose convergence_achieved flags are set explicitly."""
    scf_body = "<n_scf_steps>12</n_scf_steps><scf_error>1.0e-03</scf_error>"
    if scf_flag is not None:
        scf_body = f"<convergence_achieved>{scf_flag}</convergence_achieved>" + scf_body
    blocks = f"<scf_conv>{scf_body}</scf_conv>"
    if opt_flag is not None:
        blocks += (
            f"<opt_conv><convergence_achieved>{opt_flag}</convergence_achieved>"
            "<n_opt_steps>7</n_opt_steps></opt_conv>"
        )
    return f"{_XML_HEAD}<output><convergence_info>{blocks}</convergence_info></output></qes:espresso>"


class TestXMLConvergenceFlags(unittest.TestCase):
    """The presence of <scf_conv> means an SCF was attempted, not that it converged."""

    def test_failed_scf_is_not_reported_converged(self):
        out = parse_qe_xml(_xml_with_convergence("false"))
        self.assertIs(out.scf_converged, False)

    def test_converged_scf_is_reported_converged(self):
        out = parse_qe_xml(_xml_with_convergence("true"))
        self.assertIs(out.scf_converged, True)

    def test_failed_geometry_optimization_is_not_reported_converged(self):
        out = parse_qe_xml(_xml_with_convergence("true", opt_flag="false"))
        self.assertIs(out.opt_converged, False)

    def test_missing_flag_falls_back_to_element_presence(self):
        """Files that omit convergence_achieved keep the older, weaker signal."""
        out = parse_qe_xml(_xml_with_convergence(None))
        self.assertIs(out.scf_converged, True)

    def test_absent_convergence_info_is_unknown_not_false(self):
        """None distinguishes 'no record' from 'recorded as failed'."""
        out = parse_qe_xml(f"{_XML_HEAD}<output/></qes:espresso>")
        self.assertIsNone(out.scf_converged)


class TestXMLStressUnitsAndSign(unittest.TestCase):
    """Stress conversion must match QE's own kbar convention.

    QE prints P = +Tr(sigma)/3, and converts Ry/Bohr^3 to kbar with
    uakbar = 147105.0 -- hence 294210.26 kbar per Hartree/Bohr^3. Both are
    confirmed by the si_vc_relax.out fixture, whose first stress block prints
    diagonal (15.00, 15.15, 15.45) kbar alongside P= 15.20.
    """

    STRESS_XML = (
        f"{_XML_HEAD}<output><stress units='Hartree/Bohr**3'>\n"
        "-1.000000e-04 0.000000e+00 0.000000e+00\n"
        "0.000000e+00 -1.000000e-04 0.000000e+00\n"
        "0.000000e+00 0.000000e+00 -1.000000e-04\n"
        "</stress></output></qes:espresso>"
    )

    def setUp(self):
        self.out = parse_qe_xml(self.STRESS_XML)

    def test_stress_magnitude_in_kbar(self):
        # 1e-4 Ha/Bohr^3 * 294210.2648 kbar = 29.42102648 kbar
        self.assertAlmostEqual(self.out.stress_kbar[0][0], -29.42102648, places=6)

    def test_pressure_sign_follows_qe_convention(self):
        """An all-compressive (negative) diagonal gives a negative pressure."""
        self.assertAlmostEqual(self.out.pressure_kbar, -29.42102648, places=6)

    def test_pressure_is_positive_trace_over_three(self):
        self.assertAlmostEqual(
            self.out.pressure_kbar,
            sum(self.out.stress_kbar[i][i] for i in range(3)) / 3.0,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
