"""FCIDUMP interchange tests, including the Hartree boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum.active_space import select_active_space
from qeanalyzer.quantum.fcidump import parse_fcidump, read_fcidump, write_fcidump
from qeanalyzer.quantum.hamiltonian import build_band_model_hamiltonian, build_hubbard_hamiltonian
from qeanalyzer.quantum.units import HARTREE_TO_EV

FIXTURES = Path(__file__).parent / "fixtures"


class TestFCIDUMP(unittest.TestCase):
    def test_hubbard_fcidump_converts_ev_to_hartree(self):
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): 1.2, (1, 0): 1.2},
            onsite_u=[4.0, 4.0],
            intersite_v={(0, 1): 0.5},
            constant=-5.5,
            energy_unit="eV",
        )
        reconstructed = parse_fcidump(write_fcidump(ham))
        self.assertEqual(reconstructed.energy_unit, "Hartree")
        self.assertAlmostEqual(reconstructed.constant, -5.5 / HARTREE_TO_EV, places=12)
        self.assertAlmostEqual(reconstructed.h1[0][1], ham.h1[0][1] / HARTREE_TO_EV, places=12)
        self.assertAlmostEqual(reconstructed.h2[0][0][0][0], 4.0 / HARTREE_TO_EV, places=12)

    def test_hartree_roundtrip(self):
        ham = build_hubbard_hamiltonian(
            n_orbitals=3,
            n_electrons=3.0,
            hopping_t={(0, 1): -1.0, (1, 0): -1.0, (1, 2): -1.0, (2, 1): -1.0},
            onsite_u=3.0,
            constant=1.234,
            energy_unit="Hartree",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "FCIDUMP"
            write_fcidump(ham, path=path, orbsym=[1, 2, 1])
            loaded = read_fcidump(path)
        self.assertEqual(loaded.energy_unit, "Hartree")
        self.assertAlmostEqual(loaded.constant, ham.constant)
        self.assertTrue(loaded.is_hermitian())

    def test_band_model_export_is_explicitly_heuristic(self):
        xml = read_qe_xml(FIXTURES / "si_scf.xml")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        result = build_run_result(pw_out=pw_out, qe_xml=xml, run_id="si-scf")
        active = select_active_space(result, method="band_index", band_start=1, band_end=2)
        ham = build_band_model_hamiltonian(result, active, onsite_u_ev=2.5)
        self.assertEqual(ham.metadata["model_kind"], "qe_band_heuristic")
        self.assertFalse(ham.metadata["ab_initio"])
        loaded = parse_fcidump(write_fcidump(ham))
        self.assertEqual(loaded.n_orbitals, 2)
        self.assertAlmostEqual(loaded.h2[0][0][0][0], 2.5 / HARTREE_TO_EV, places=10)

    def test_fractional_electron_sector_is_rejected(self):
        ham = build_hubbard_hamiltonian(2, 1.6, onsite_u=2.0)
        with self.assertRaisesRegex(ValueError, "fractional"):
            write_fcidump(ham)

    def test_errors(self):
        with self.assertRaises(ValueError):
            parse_fcidump("")
        with self.assertRaises(ValueError):
            parse_fcidump("&FCI NELEC=2 /")
        with self.assertRaises(NotImplementedError):
            parse_fcidump("&FCI NORB=2, NELEC=2, IUHF=1 /\n0.0 0 0 0 0\n")


if __name__ == "__main__":
    unittest.main()
