"""Tests for MaterialHamiltonian and Hamiltonian construction (src/qeanalyzer/quantum/hamiltonian.py)."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum.active_space import select_active_space
from qeanalyzer.quantum.hamiltonian import (
    MaterialHamiltonian,
    build_active_space_hamiltonian,
    build_hubbard_hamiltonian,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestMaterialHamiltonian(unittest.TestCase):
    """Test MaterialHamiltonian data structure, properties, and symmetries."""

    def test_initialization_defaults(self):
        ham = MaterialHamiltonian(n_orbitals=3, n_electrons=4.0, constant=1.5)
        self.assertEqual(ham.n_orbitals, 3)
        self.assertEqual(ham.n_spin_orbitals, 6)
        self.assertEqual(len(ham.h1), 3)
        self.assertEqual(len(ham.h1[0]), 3)
        self.assertEqual(len(ham.h2), 3)
        self.assertEqual(len(ham.h2[0][0][0]), 3)
        self.assertEqual(ham.constant, 1.5)
        self.assertTrue(ham.is_hermitian())
        self.assertIn("Spatial Orbitals  : 3", ham.summary())

    def test_invalid_n_orbitals_raises(self):
        with self.assertRaises(ValueError):
            MaterialHamiltonian(n_orbitals=0, n_electrons=2.0)

    def test_hermiticity_check(self):
        # 1. Valid symmetric Hamiltonian
        ham = MaterialHamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            h1=[[1.0, -0.5], [-0.5, 2.0]],
        )
        self.assertTrue(ham.is_hermitian())

        # 2. Asymmetric 1-body matrix
        ham_asym = MaterialHamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            h1=[[1.0, -0.5], [0.5, 2.0]],
        )
        self.assertFalse(ham_asym.is_hermitian())

    def test_serialization_roundtrip(self):
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): -1.0, (1, 0): -1.0},
            onsite_u=[3.5, 3.5],
            intersite_v={(0, 1): 0.8, (1, 0): 0.8},
            constant=-10.5,
        )
        d = ham.to_dict()
        reconstructed = MaterialHamiltonian.from_dict(d)

        self.assertEqual(reconstructed.n_orbitals, 2)
        self.assertEqual(reconstructed.n_spin_orbitals, 4)
        self.assertAlmostEqual(reconstructed.constant, -10.5)
        self.assertEqual(reconstructed.h1, ham.h1)
        self.assertEqual(reconstructed.h2, ham.h2)
        self.assertEqual(reconstructed.onsite_u, [3.5, 3.5])
        self.assertAlmostEqual(reconstructed.intersite_v[(0, 1)], 0.8)


class TestHubbardHamiltonianBuilder(unittest.TestCase):
    """Test parameterized Hubbard model builder."""

    def test_two_site_hubbard_model(self):
        # 2 sites, t = 1.0, U = 4.0, V = 1.0
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): 1.0, (1, 0): 1.0},
            onsite_u=4.0,
            intersite_v={(0, 1): 1.0, (1, 0): 1.0},
        )
        self.assertEqual(ham.n_orbitals, 2)
        self.assertEqual(ham.n_spin_orbitals, 4)

        # 1-body hoppings: h1[0][1] = h1[1][0] = -t = -1.0
        self.assertAlmostEqual(ham.h1[0][1], -1.0)
        self.assertAlmostEqual(ham.h1[1][0], -1.0)
        self.assertAlmostEqual(ham.h1[0][0], 0.0)

        # 2-body onsite U: (00|00) = (11|11) = 4.0
        self.assertAlmostEqual(ham.h2[0][0][0][0], 4.0)
        self.assertAlmostEqual(ham.h2[1][1][1][1], 4.0)

        # 2-body intersite V: (00|11) = (11|00) = 1.0
        self.assertAlmostEqual(ham.h2[0][0][1][1], 1.0)
        self.assertAlmostEqual(ham.h2[1][1][0][0], 1.0)

        self.assertTrue(ham.is_hermitian())
        n1, n2 = ham.count_nonzero_integrals()
        self.assertEqual(n1, 2)
        self.assertEqual(n2, 4)

    def test_hubbard_from_matrix_hopping(self):
        # h1 follows H = -t sum_<ij> c_i^+ c_j, so an off-diagonal t enters as -t.
        # The caller's sign is preserved rather than normalised away: t = -1.5 and
        # t = +1.5 are distinct models and must not produce the same h1.
        t_matrix = [[0.0, -1.5], [-1.5, 0.0]]
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t=t_matrix,
            onsite_u=[2.0, 3.0],
        )
        self.assertAlmostEqual(ham.h1[0][1], 1.5)
        self.assertAlmostEqual(ham.onsite_u[0], 2.0)
        self.assertAlmostEqual(ham.onsite_u[1], 3.0)

        flipped = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t=[[0.0, 1.5], [1.5, 0.0]],
            onsite_u=[2.0, 3.0],
        )
        self.assertAlmostEqual(flipped.h1[0][1], -1.5)


class TestActiveSpaceHamiltonianFromDFT(unittest.TestCase):
    """Test constructing MaterialHamiltonian from DFT calculation and ActiveSpace."""

    @classmethod
    def setUpClass(cls):
        cls.xml = read_qe_xml(FIXTURES / "si_scf.xml")
        cls.pw_out = read_pw_output(FIXTURES / "si_scf.out")
        cls.result = build_run_result(pw_out=cls.pw_out, qe_xml=cls.xml, run_id="si-scf")

    def test_build_active_space_hamiltonian_silicon(self):
        # Select active space for bands 1 and 2
        asp = select_active_space(self.result, method="band_index", band_start=1, band_end=2)
        self.assertEqual(asp.n_active_orbitals, 2)

        ham = build_active_space_hamiltonian(
            state=self.result,
            active_space=asp,
            onsite_u_ev=3.0,
            intersite_v_ev=0.5,
        )

        self.assertEqual(ham.n_orbitals, 2)
        self.assertEqual(ham.n_spin_orbitals, 4)
        self.assertAlmostEqual(ham.n_electrons, asp.n_active_electrons)
        self.assertTrue(ham.is_hermitian())

        # 1-body diagonal must reflect DFT eigenvalues
        self.assertNotEqual(ham.h1[0][0], 0.0)
        self.assertNotEqual(ham.h1[1][1], 0.0)

        # 2-body interaction
        self.assertAlmostEqual(ham.h2[0][0][0][0], 3.0)
        self.assertAlmostEqual(ham.h2[1][1][1][1], 3.0)
        self.assertAlmostEqual(ham.h2[0][0][1][1], 0.5)

        # Core electrons energy shift is non-zero because band 0 is frozen core
        self.assertNotEqual(ham.constant, 0.0)


class TestSpinOrbitalConversion(unittest.TestCase):
    """Test conversion from spatial orbital integrals to spin-orbital basis."""

    def test_spin_orbital_anti_symmetrization(self):
        ham = build_hubbard_hamiltonian(
            n_orbitals=2,
            n_electrons=2.0,
            hopping_t={(0, 1): 1.0, (1, 0): 1.0},
            onsite_u=4.0,
        )
        h1_so, h2_so = ham.to_spin_orbital_integrals()

        # Check dimensions: 2*norb = 4
        self.assertEqual(len(h1_so), 4)
        self.assertEqual(len(h2_so), 4)

        # Check 1-body spin conservation:
        # Spatial (0, 1) corresponds to alpha (0, 2) and beta (1, 3)
        self.assertAlmostEqual(h1_so[0][2], -1.0)  # 0_alpha -> 1_alpha
        self.assertAlmostEqual(h1_so[1][3], -1.0)  # 0_beta  -> 1_beta
        self.assertAlmostEqual(h1_so[0][3], 0.0)   # Spin-flip is 0

        # Check 2-body onsite repulsion in spin-orbital basis:
        # Spatial (00|00) with alpha and beta has <0_a, 0_b | 0_a, 0_b> = U = 4.0
        self.assertAlmostEqual(h2_so[0][1][0][1], 4.0)
        self.assertAlmostEqual(h2_so[1][0][1][0], 4.0)

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
