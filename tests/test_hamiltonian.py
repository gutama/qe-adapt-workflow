"""Tests for the finite Hamiltonian convention and model builders."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum.active_space import select_active_space
from qeanalyzer.quantum.hamiltonian import (
    MaterialHamiltonian,
    build_band_model_hamiltonian,
    build_hubbard_hamiltonian,
    build_integral_hamiltonian,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestMaterialHamiltonian(unittest.TestCase):
    def test_defaults_and_convention(self):
        ham = MaterialHamiltonian(3, 4.0, constant=1.5)
        self.assertEqual(ham.n_spin_orbitals, 6)
        self.assertTrue(ham.is_hermitian())
        self.assertEqual(ham.to_dict()["integral_convention"], "chemist_(pq|rs)")

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            MaterialHamiltonian(0, 2.0)
        with self.assertRaises(ValueError):
            MaterialHamiltonian(2, 2.0, h1=[[1.0]])

    def test_serialization(self):
        ham = build_hubbard_hamiltonian(
            2, 2.0,
            hopping_t={(0, 1): -1.0, (1, 0): -1.0},
            onsite_u=[3.5, 3.5],
            intersite_v={(0, 1): 0.8},
            constant=-10.5,
        )
        reconstructed = MaterialHamiltonian.from_dict(ham.to_dict())
        self.assertEqual(reconstructed.h1, ham.h1)
        self.assertEqual(reconstructed.h2, ham.h2)
        self.assertEqual(reconstructed.energy_unit, "eV")

    def test_explicit_integral_builder_checks_symmetry(self):
        h1 = [[0.0, -1.0], [-1.0, 0.0]]
        h2 = [[[[0.0 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)]
        h2[0][0][0][0] = 2.0
        h2[1][1][1][1] = 2.0
        ham = build_integral_hamiltonian(h1, h2, n_electrons=2, energy_unit="Hartree")
        self.assertTrue(ham.is_hermitian())
        self.assertEqual(ham.metadata["model_kind"], "explicit_integrals")


class TestHubbardHamiltonian(unittest.TestCase):
    def test_sign_and_interactions(self):
        ham = build_hubbard_hamiltonian(
            2, 2.0,
            hopping_t={(0, 1): 1.0, (1, 0): 1.0},
            onsite_u=4.0,
            intersite_v={(0, 1): 1.0},
        )
        self.assertEqual(ham.h1[0][1], -1.0)
        self.assertEqual(ham.h2[0][0][0][0], 4.0)
        self.assertEqual(ham.h2[0][0][1][1], 1.0)
        self.assertEqual(ham.h2[1][1][0][0], 1.0)
        self.assertTrue(ham.is_hermitian())

    def test_asymmetric_hopping_rejected(self):
        with self.assertRaises(ValueError):
            build_hubbard_hamiltonian(2, 2.0, hopping_t={(0, 1): 1.0})

    def test_hopping_sign_preserved(self):
        positive = build_hubbard_hamiltonian(2, 2.0, {(0, 1): 1.0, (1, 0): 1.0})
        negative = build_hubbard_hamiltonian(2, 2.0, {(0, 1): -1.0, (1, 0): -1.0})
        self.assertEqual(positive.h1[0][1], -1.0)
        self.assertEqual(negative.h1[0][1], 1.0)

    def test_spin_orbital_helper(self):
        ham = build_hubbard_hamiltonian(2, 2.0, {(0, 1): 1.0, (1, 0): 1.0}, onsite_u=4.0)
        h1so, h2so = ham.to_spin_orbital_integrals()
        self.assertEqual(h1so[0][2], -1.0)
        self.assertEqual(h1so[1][3], -1.0)
        self.assertEqual(h2so[0][1][0][1], 4.0)


class TestBandModelBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xml = read_qe_xml(FIXTURES / "si_scf.xml")
        out = read_pw_output(FIXTURES / "si_scf.out")
        cls.result = build_run_result(pw_out=out, qe_xml=xml, run_id="si")

    def test_band_model_is_not_claimed_ab_initio(self):
        active = select_active_space(self.result, method="band_index", band_start=1, band_end=2)
        ham = build_band_model_hamiltonian(self.result, active, onsite_u_ev=3.0, intersite_v_ev=0.5)
        self.assertEqual(ham.metadata["model_kind"], "qe_band_heuristic")
        self.assertEqual(ham.metadata["scientific_status"], "experimental_heuristic")
        self.assertFalse(ham.metadata["ab_initio"])
        self.assertEqual(ham.constant, 0.0)
        self.assertIn("not an ab-initio", ham.metadata["warning"].lower())


class TestHoppingSignIsPreserved(unittest.TestCase):
    """t = +1 and t = -1 are distinct models and must not collapse onto one h1.

    Negating only positive hoppings mapped both onto the same matrix, so the
    sign a caller supplied could not be recovered and the stored hopping_t dict
    disagreed with h1 for negative t.
    """

    @staticmethod
    def _h1_for(t: float) -> float:
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): t, (1, 0): t},
            onsite_u=4.0,
        )
        return ham.h1[0][1]

    def test_opposite_signs_give_opposite_matrices(self):
        self.assertAlmostEqual(self._h1_for(1.0), -1.0)
        self.assertAlmostEqual(self._h1_for(-1.0), 1.0)
        self.assertNotAlmostEqual(self._h1_for(1.0), self._h1_for(-1.0))

    def test_convention_is_applied_unconditionally(self):
        for t in (0.25, 1.0, 2.5, -0.25, -1.0, -2.5):
            with self.subTest(t=t):
                self.assertAlmostEqual(self._h1_for(t), -t)

    def test_diagonal_entries_are_onsite_energies(self):
        """Diagonal terms are site energies and pass through unnegated."""
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 0): 0.75, (1, 1): -0.5},
            onsite_u=4.0,
        )
        self.assertAlmostEqual(ham.h1[0][0], 0.75)
        self.assertAlmostEqual(ham.h1[1][1], -0.5)

    def test_hermiticity_is_retained(self):
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): -1.3, (1, 0): -1.3},
            onsite_u=4.0,
        )
        self.assertTrue(ham.is_hermitian())


if __name__ == "__main__":
    unittest.main()
