"""Tests for canonical models (src/qeanalyzer/models)."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import (
    QEAtom,
    QEConvergenceHistory,
    QEElectronicState,
    QEProvenance,
    QERunResult,
    QERunStatus,
    QEStructure,
    build_run_result,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Provenance Model Tests
# ---------------------------------------------------------------------------

class TestProvenanceModel(unittest.TestCase):
    """Test QEProvenance creation, hashing, and dict roundtrips."""

    def test_compute_sha256(self):
        digest = QEProvenance.compute_sha256("test string")
        self.assertEqual(len(digest), 64)
        self.assertIsInstance(digest, str)

    def test_to_dict_from_dict_roundtrip(self):
        prov = QEProvenance(
            run_id="run-001",
            parent_run="run-000",
            qe_version="7.2",
            input_sha256="abc123hash",
            pseudo_sha256={"Si": "ps123"},
            prefix="silicon",
            outdir="./tmp",
            n_mpi_processes=4,
            n_threads=1,
            policy_name="scf-to-nscf",
            policy_version="1.0",
        )
        d = prov.to_dict()
        reconstructed = QEProvenance.from_dict(d)

        self.assertEqual(reconstructed.run_id, "run-001")
        self.assertEqual(reconstructed.parent_run, "run-000")
        self.assertEqual(reconstructed.qe_version, "7.2")
        self.assertEqual(reconstructed.input_sha256, "abc123hash")
        self.assertEqual(reconstructed.pseudo_sha256, {"Si": "ps123"})
        self.assertEqual(reconstructed.n_mpi_processes, 4)
        self.assertEqual(reconstructed.policy_name, "scf-to-nscf")


# ---------------------------------------------------------------------------
# Structure Model Tests
# ---------------------------------------------------------------------------

class TestStructureModel(unittest.TestCase):
    """Test QEStructure and QEAtom properties and conversions."""

    def test_atom_from_bohr(self):
        atom = QEAtom.from_bohr("Si", (0.0, 5.1, 5.1), (0.0, 0.0, 0.01))
        self.assertEqual(atom.symbol, "Si")
        self.assertEqual(atom.position_bohr, (0.0, 5.1, 5.1))
        self.assertAlmostEqual(atom.position_angstrom[1], 5.1 * 0.529177210903)
        self.assertEqual(atom.force_ry_bohr, (0.0, 0.0, 0.01))

    def test_structure_volume_and_forces(self):
        cell_bohr = (
            (5.0, 0.0, 0.0),
            (0.0, 5.0, 0.0),
            (0.0, 0.0, 5.0),
        )
        atoms = [
            ("Si", (0.0, 0.0, 0.0)),
            ("Si", (2.5, 2.5, 2.5)),
        ]
        forces = [
            (0.0, 0.0, 0.05),
            (0.0, 0.0, -0.05),
        ]
        struct = QEStructure.from_cell_and_atoms_bohr(
            cell_bohr=cell_bohr,
            atoms_bohr=atoms,
            alat_bohr=10.0,
            forces_ry_bohr=forces,
        )

        self.assertAlmostEqual(struct.volume_bohr3, 125.0)
        self.assertAlmostEqual(struct.max_force_ry_bohr, 0.05)
        self.assertEqual(len(struct.atoms), 2)

    def test_structure_dict_roundtrip(self):
        cell_bohr = (
            (5.0, 0.0, 0.0),
            (0.0, 5.0, 0.0),
            (0.0, 0.0, 5.0),
        )
        struct = QEStructure.from_cell_and_atoms_bohr(
            cell_bohr=cell_bohr,
            atoms_bohr=[("Si", (0.0, 0.0, 0.0))],
            alat_bohr=10.0,
        )
        d = struct.to_dict()
        reconstructed = QEStructure.from_dict(d)
        self.assertAlmostEqual(reconstructed.alat_bohr, 10.0)
        self.assertEqual(reconstructed.cell_vectors_bohr, cell_bohr)


# ---------------------------------------------------------------------------
# Electronic State Tests
# ---------------------------------------------------------------------------

class TestElectronicStateModel(unittest.TestCase):
    """Test QEElectronicState properties and conversions."""

    def test_energy_and_band_gap_derivations(self):
        state = QEElectronicState.from_energies(
            total_energy_ry=-15.85434567,
            highest_occupied_ev=6.2140,
            lowest_unoccupied_ev=7.3940,
            n_electrons=8.0,
            n_bands=4,
        )
        self.assertAlmostEqual(state.total_energy_ry, -15.85434567)
        self.assertIsNotNone(state.total_energy_hartree)
        self.assertAlmostEqual(state.band_gap_ev, 1.18)
        self.assertFalse(state.is_metal)

    def test_metallic_state_detection(self):
        state = QEElectronicState.from_energies(
            total_energy_ry=-4.18693542,
            fermi_energy_ev=7.9421,
            n_electrons=3.0,
        )
        self.assertTrue(state.is_metal)
        self.assertIsNone(state.band_gap_ev)

    def test_electronic_state_dict_roundtrip(self):
        state = QEElectronicState.from_energies(
            total_energy_ry=-10.0,
            one_electron_ry=2.0,
            fermi_energy_ev=5.0,
            eigenvalues_ev=[[-1.0, 0.0, 2.0]],
            occupations=[[2.0, 1.0, 0.0]],
        )
        d = state.to_dict()
        reconstructed = QEElectronicState.from_dict(d)
        self.assertAlmostEqual(reconstructed.total_energy_ry, -10.0)
        self.assertEqual(reconstructed.eigenvalues_ev, [[-1.0, 0.0, 2.0]])


# ---------------------------------------------------------------------------
# Full QERunResult and build_run_result Tests
# ---------------------------------------------------------------------------

class TestBuildRunResult(unittest.TestCase):
    """Test unified run result construction across XML, pw.out, and pw.in."""

    def test_build_silicon_scf(self):
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")

        result = build_run_result(
            pw_in=pw_in,
            pw_out=pw_out,
            qe_xml=qe_xml,
            run_id="si-001",
            input_text=(FIXTURES / "si_scf.in").read_text(),
        )

        self.assertEqual(result.run_id, "si-001")
        self.assertEqual(result.calculation, "scf")
        self.assertTrue(result.status.completed)
        self.assertTrue(result.status.scf_converged)
        self.assertEqual(result.status.exit_status, "converged")

        # Electronic ground truth
        self.assertAlmostEqual(result.electronic.total_energy_ry, -15.85434567)
        self.assertAlmostEqual(result.electronic.band_gap_ev, 1.18, delta=0.01)
        self.assertEqual(result.electronic.n_bands, 4)

        # Structure
        self.assertIsNotNone(result.structure)
        self.assertEqual(len(result.structure.atoms), 2)
        self.assertEqual(result.structure.atoms[0].symbol, "Si")

        # Convergence
        self.assertEqual(result.convergence.n_scf_steps, 8)
        self.assertEqual(len(result.convergence.scf_iterations), 8)
        self.assertAlmostEqual(result.convergence.wall_time_seconds, 0.54)

    def test_build_aluminum_metal(self):
        pw_in = read_pw_input(FIXTURES / "al_scf_metal.in")
        pw_out = read_pw_output(FIXTURES / "al_scf_metal.out")
        qe_xml = read_qe_xml(FIXTURES / "al_scf_metal.xml")

        result = build_run_result(
            pw_in=pw_in,
            pw_out=pw_out,
            qe_xml=qe_xml,
            run_id="al-001",
        )

        self.assertTrue(result.status.scf_converged)
        self.assertAlmostEqual(result.electronic.fermi_energy_ev, 7.9421, delta=0.01)
        self.assertTrue(result.electronic.is_metal)

    def test_build_silicon_vc_relax(self):
        pw_in = read_pw_input(FIXTURES / "si_vc_relax.in")
        pw_out = read_pw_output(FIXTURES / "si_vc_relax.out")
        qe_xml = read_qe_xml(FIXTURES / "si_vc_relax.xml")

        result = build_run_result(
            pw_in=pw_in,
            pw_out=pw_out,
            qe_xml=qe_xml,
            run_id="si-vcrelax-001",
        )

        self.assertEqual(result.calculation, "vc-relax")
        self.assertTrue(result.status.opt_converged)
        self.assertIsNotNone(result.initial_structure)
        self.assertIsNotNone(result.structure)
        self.assertEqual(len(result.convergence.ionic_steps), 2)

    def test_run_result_dict_roundtrip(self):
        pw_in = read_pw_input(FIXTURES / "si_scf.in")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")

        result = build_run_result(
            pw_in=pw_in,
            pw_out=pw_out,
            qe_xml=qe_xml,
            run_id="si-001",
            parent_run="root",
        )

        d = result.to_dict()
        self.assertEqual(d["schema_version"], "1.0")
        self.assertEqual(d["run_id"], "si-001")
        self.assertEqual(d["parent_run"], "root")

        reconstructed = QERunResult.from_dict(d)
        self.assertEqual(reconstructed.run_id, "si-001")
        self.assertEqual(reconstructed.calculation, "scf")
        self.assertTrue(reconstructed.status.scf_converged)
        self.assertAlmostEqual(reconstructed.electronic.total_energy_ry, -15.85434567)


if __name__ == "__main__":
    unittest.main()
