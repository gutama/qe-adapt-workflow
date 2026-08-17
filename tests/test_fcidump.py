"""Tests for FCIDUMP writer, parser, and file IO (src/qeanalyzer/quantum/fcidump.py)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum.active_space import select_active_space
from qeanalyzer.quantum.fcidump import parse_fcidump, read_fcidump, write_fcidump
from qeanalyzer.quantum.hamiltonian import (
    MaterialHamiltonian,
    build_active_space_hamiltonian,
    build_hubbard_hamiltonian,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestFCIDUMP(unittest.TestCase):
    """Test FCIDUMP serialization, parsing, and round-trip preservation."""

    def test_hubbard_fcidump_roundtrip(self):
        # 2-site Hubbard model with hopping, onsite U, and intersite V
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): 1.2, (1, 0): 1.2},
            onsite_u=[4.0, 4.0],
            intersite_v={(0, 1): 0.5, (1, 0): 0.5},
            constant=-5.5,
        )

        fci_text = write_fcidump(ham)
        self.assertIn("&FCI", fci_text)
        self.assertIn("NORB=2", fci_text)
        self.assertIn("NELEC=2", fci_text)

        reconstructed = parse_fcidump(fci_text)
        self.assertEqual(reconstructed.n_orbitals, 2)
        self.assertEqual(reconstructed.n_spin_orbitals, 4)
        self.assertAlmostEqual(reconstructed.n_electrons, 2.0)
        self.assertAlmostEqual(reconstructed.constant, -5.5)

        # Verify 1-body integrals
        for p in range(2):
            for q in range(2):
                self.assertAlmostEqual(reconstructed.h1[p][q], ham.h1[p][q], places=10)

        # Verify 2-body integrals
        for p in range(2):
            for q in range(2):
                for r in range(2):
                    for s in range(2):
                        self.assertAlmostEqual(
                            reconstructed.h2[p][q][r][s],
                            ham.h2[p][q][r][s],
                            places=10,
                        )

    def test_file_io_roundtrip(self):
        ham = build_hubbard_hamiltonian(
            n_orbitals=3,
            n_electrons=3.0,
            hopping_t={(0, 1): -1.0, (1, 2): -1.0},
            onsite_u=3.0,
            constant=1.234,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fci_file = Path(tmpdir) / "model.fcidump"
            write_fcidump(ham, path=fci_file, orbsym=[1, 2, 1], isym=1)
            self.assertTrue(fci_file.exists())
            self.assertGreater(fci_file.stat().st_size, 100)

            loaded = read_fcidump(fci_file)
            self.assertEqual(loaded.n_orbitals, 3)
            self.assertAlmostEqual(loaded.constant, 1.234)
            self.assertTrue(loaded.is_hermitian())

    def test_dft_active_space_fcidump_silicon(self):
        xml = read_qe_xml(FIXTURES / "si_scf.xml")
        pw_out = read_pw_output(FIXTURES / "si_scf.out")
        result = build_run_result(pw_out=pw_out, qe_xml=xml, run_id="si-scf")

        asp = select_active_space(result, method="band_index", band_start=1, band_end=2)
        ham = build_active_space_hamiltonian(result, active_space=asp, onsite_u_ev=2.5)

        text = write_fcidump(ham)
        reconstructed = parse_fcidump(text)

        self.assertEqual(reconstructed.n_orbitals, 2)
        self.assertAlmostEqual(reconstructed.n_electrons, asp.n_active_electrons)
        self.assertAlmostEqual(reconstructed.constant, ham.constant, places=6)
        self.assertAlmostEqual(reconstructed.h1[0][0], ham.h1[0][0], places=6)
        self.assertAlmostEqual(reconstructed.h1[1][1], ham.h1[1][1], places=6)
        self.assertAlmostEqual(reconstructed.h2[0][0][0][0], 2.5, places=6)

    def test_edge_cases_and_errors(self):
        with self.assertRaises(ValueError):
            parse_fcidump("")

        with self.assertRaises(ValueError):
            parse_fcidump("&FCI NELEC=2 /")  # Missing NORB


if __name__ == "__main__":
    unittest.main()
