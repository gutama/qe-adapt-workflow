"""Tests for active space selection strategies (src/qeanalyzer/quantum/active_space.py)."""

from __future__ import annotations

import unittest
from pathlib import Path

from qeanalyzer.io import read_pw_output, read_qe_xml
from qeanalyzer.models import QEElectronicState, QERunResult, build_run_result
from qeanalyzer.quantum.active_space import (
    ActiveSpace,
    BandIndexSelector,
    EnergyWindowSelector,
    ExplicitOrbitalSelector,
    OccupationSelector,
    create_active_space_selector,
    select_active_space,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestActiveSpace(unittest.TestCase):
    """Test ActiveSpace data structure and serialization."""

    def test_active_space_defaults_and_post_init(self):
        asp = ActiveSpace(
            method="band_index",
            active_orbitals=[3, 1, 2],  # Unordered
            n_active_electrons=4.0,
            frozen_core_orbitals=[0],
            frozen_virtual_orbitals=[5, 4],
        )
        self.assertEqual(asp.active_orbitals, [1, 2, 3])
        self.assertEqual(asp.n_active_orbitals, 3)
        self.assertEqual(asp.n_spin_orbitals, 6)
        self.assertEqual(asp.frozen_virtual_orbitals, [4, 5])
        self.assertIn("Bands 1..3", asp.summary())

    def test_active_space_serialization_roundtrip(self):
        asp = ActiveSpace(
            method="energy_window",
            active_orbitals=[2, 3, 4],
            n_active_orbitals=3,
            n_spin_orbitals=6,
            n_active_electrons=4.0,
            frozen_core_orbitals=[0, 1],
            frozen_virtual_orbitals=[5, 6, 7],
            n_core_electrons=4.0,
            energy_window_ev=(-2.5, 2.5),
            fermi_energy_ev=5.12,
            metadata={"test_key": "test_val"},
        )
        d = asp.to_dict()
        reconstructed = ActiveSpace.from_dict(d)
        self.assertEqual(reconstructed.method, "energy_window")
        self.assertEqual(reconstructed.active_orbitals, [2, 3, 4])
        self.assertEqual(reconstructed.n_active_orbitals, 3)
        self.assertEqual(reconstructed.n_spin_orbitals, 6)
        self.assertAlmostEqual(reconstructed.n_active_electrons, 4.0)
        self.assertEqual(reconstructed.energy_window_ev, (-2.5, 2.5))
        self.assertAlmostEqual(reconstructed.fermi_energy_ev, 5.12)
        self.assertEqual(reconstructed.metadata["test_key"], "test_val")


class TestEnergyWindowSelector(unittest.TestCase):
    """Test EnergyWindowSelector on Silicon SCF calculation."""

    @classmethod
    def setUpClass(cls):
        cls.xml = read_qe_xml(FIXTURES / "si_scf.xml")
        cls.pw_out = read_pw_output(FIXTURES / "si_scf.out")
        cls.result = build_run_result(pw_out=cls.pw_out, qe_xml=cls.xml, run_id="si-scf")

    def test_select_relative_to_fermi(self):
        selector = EnergyWindowSelector(emin_ev=-8.0, emax_ev=1.0, relative_to_fermi=True)
        asp = selector.select(self.result)

        self.assertEqual(asp.method, "energy_window")
        self.assertGreater(asp.n_active_orbitals, 0)
        self.assertEqual(asp.n_spin_orbitals, asp.n_active_orbitals * 2)
        self.assertGreater(asp.n_active_electrons, 0.0)
        self.assertIsNotNone(asp.fermi_energy_ev)

        # Check disjoint orbital sets
        all_selected = set(asp.active_orbitals)
        self.assertTrue(all_selected.isdisjoint(set(asp.frozen_core_orbitals)))
        self.assertTrue(all_selected.isdisjoint(set(asp.frozen_virtual_orbitals)))

    def test_select_absolute_energy_window(self):
        # Silicon valence bands are between -6.0 eV and +6.5 eV
        selector = EnergyWindowSelector(emin_ev=-2.0, emax_ev=7.0, relative_to_fermi=False)
        asp = selector.select(self.result)

        self.assertEqual(asp.method, "energy_window")
        self.assertGreater(asp.n_active_orbitals, 0)
        self.assertEqual(asp.energy_window_ev, (-2.0, 7.0))

    def test_kpoint_mode_any_and_average(self):
        sel_avg = EnergyWindowSelector(emin_ev=-8.0, emax_ev=1.0, kpoint_mode="average")
        asp_avg = sel_avg.select(self.result)
        self.assertGreater(asp_avg.n_active_orbitals, 0)

        sel_any = EnergyWindowSelector(emin_ev=-8.0, emax_ev=1.0, kpoint_mode="any")
        asp_any = sel_any.select(self.result)
        self.assertGreaterEqual(asp_any.n_active_orbitals, asp_avg.n_active_orbitals)

    def test_invalid_window_bounds_raises(self):
        with self.assertRaises(ValueError):
            EnergyWindowSelector(emin_ev=5.0, emax_ev=2.0)

    def test_no_bands_in_window_raises(self):
        selector = EnergyWindowSelector(emin_ev=100.0, emax_ev=200.0, relative_to_fermi=True)
        with self.assertRaises(ValueError):
            selector.select(self.result)


class TestBandIndexSelector(unittest.TestCase):
    """Test BandIndexSelector with explicit ranges and indices."""

    @classmethod
    def setUpClass(cls):
        cls.xml = read_qe_xml(FIXTURES / "si_scf.xml")
        cls.result = build_run_result(qe_xml=cls.xml, run_id="si-scf")

    def test_select_by_list(self):
        selector = BandIndexSelector(band_indices=[1, 2, 3])
        asp = selector.select(self.result)

        self.assertEqual(asp.active_orbitals, [1, 2, 3])
        self.assertEqual(asp.n_active_orbitals, 3)
        self.assertEqual(asp.n_spin_orbitals, 6)
        self.assertEqual(asp.frozen_core_orbitals, [0])
        self.assertEqual(asp.frozen_virtual_orbitals, [])

    def test_select_by_start_end(self):
        selector = BandIndexSelector(band_start=1, band_end=2)
        asp = selector.select(self.result)

        self.assertEqual(asp.active_orbitals, [1, 2])
        self.assertEqual(asp.frozen_core_orbitals, [0])
        self.assertEqual(asp.frozen_virtual_orbitals, [3])

    def test_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            BandIndexSelector(band_start=5, band_end=2)

    def test_out_of_bounds_raises(self):
        selector = BandIndexSelector(band_indices=[100, 101])
        with self.assertRaises(IndexError):
            selector.select(self.result)


class TestOccupationSelector(unittest.TestCase):
    """Test OccupationSelector on Aluminum metal SCF calculation."""

    @classmethod
    def setUpClass(cls):
        cls.xml = read_qe_xml(FIXTURES / "al_scf_metal.xml")
        cls.result = build_run_result(qe_xml=cls.xml, run_id="al-metal")

    def test_select_partially_occupied_orbitals(self):
        selector = OccupationSelector(min_occ=0.01, max_occ=1.99)
        asp = selector.select(self.result)

        self.assertEqual(asp.method, "occupation")
        self.assertGreater(asp.n_active_orbitals, 0)
        self.assertGreater(asp.n_active_electrons, 0.0)

    def test_no_occupations_raises(self):
        state = QEElectronicState(total_energy_ry=-10.0)
        selector = OccupationSelector()
        with self.assertRaises(ValueError):
            selector.select(state)


class TestExplicitOrbitalSelector(unittest.TestCase):
    """Test ExplicitOrbitalSelector for manual active-space override."""

    def test_manual_override(self):
        selector = ExplicitOrbitalSelector(
            active_orbitals=[2, 3],
            n_active_electrons=2.0,
            frozen_core_orbitals=[0, 1],
            frozen_virtual_orbitals=[4, 5],
        )
        state = QEElectronicState(fermi_energy_ev=5.0)
        asp = selector.select(state)

        self.assertEqual(asp.method, "explicit")
        self.assertEqual(asp.active_orbitals, [2, 3])
        self.assertEqual(asp.n_active_orbitals, 2)
        self.assertEqual(asp.n_spin_orbitals, 4)
        self.assertAlmostEqual(asp.n_active_electrons, 2.0)
        self.assertTrue(asp.metadata.get("manual_override"))


class TestActiveSpaceFactory(unittest.TestCase):
    """Test factory and convenience functions."""

    @classmethod
    def setUpClass(cls):
        cls.xml = read_qe_xml(FIXTURES / "si_scf.xml")
        cls.result = build_run_result(qe_xml=cls.xml, run_id="si-scf")

    def test_create_selector_by_name(self):
        sel_ew = create_active_space_selector("energy_window", emin_ev=-2.0, emax_ev=2.0)
        self.assertIsInstance(sel_ew, EnergyWindowSelector)

        sel_bi = create_active_space_selector("band_index", band_start=0, band_end=3)
        self.assertIsInstance(sel_bi, BandIndexSelector)

        sel_occ = create_active_space_selector("occupation", min_occ=0.1)
        self.assertIsInstance(sel_occ, OccupationSelector)

        sel_exp = create_active_space_selector("explicit", active_orbitals=[0, 1], n_active_electrons=2.0)
        self.assertIsInstance(sel_exp, ExplicitOrbitalSelector)

    def test_unknown_selector_raises(self):
        with self.assertRaises(ValueError):
            create_active_space_selector("unknown_method")

    def test_select_active_space_convenience(self):
        asp = select_active_space(self.result, method="band_index", band_start=1, band_end=3)
        self.assertEqual(asp.active_orbitals, [1, 2, 3])
        self.assertEqual(asp.n_active_orbitals, 3)


if __name__ == "__main__":
    unittest.main()
