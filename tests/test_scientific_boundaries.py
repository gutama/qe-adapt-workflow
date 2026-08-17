from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io.source_bundle import resolve_qe_source_paths
from qeanalyzer.models import QEElectronicState, QERunResult
from qeanalyzer.quantum import (
    BandIndexSelector,
    ExactDiagonalizationSolver,
    apply_quantum_feedback,
    select_active_space,
)
from qeanalyzer.quantum.hamiltonian import build_hubbard_hamiltonian
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult
from qeanalyzer.workflow.outer_loop import ConvergenceCriteria, OuterLoopLedger


class TestWeightedPeriodicSelection(unittest.TestCase):
    def test_occupation_count_uses_kpoint_weights(self):
        state = QEElectronicState(
            n_bands=2,
            n_kpoints=2,
            n_electrons=2.0,
            eigenvalues_ev=[[0.0, 1.0], [0.2, 1.2]],
            occupations=[[2.0, 0.0], [0.0, 2.0]],
            kpoint_weights=[0.75, 0.25],
        )
        active = BandIndexSelector(band_indices=[0]).select(state)
        self.assertAlmostEqual(active.n_active_electrons, 1.5)

    def test_spinor_results_are_not_silently_doubled(self):
        state = QEElectronicState(
            n_bands=2,
            eigenvalues_ev=[[0.0, 1.0]],
            occupations=[[1.0, 1.0]],
            spinorbit=True,
        )
        with self.assertRaises(NotImplementedError):
            BandIndexSelector(band_indices=[0]).select(state)


class TestParticleNumberBoundary(unittest.TestCase):
    def test_exact_solver_rejects_fractional_sector(self):
        ham = build_hubbard_hamiltonian(2, 1.6, hopping_t={(0, 1): 1.0, (1, 0): 1.0})
        with self.assertRaisesRegex(ValueError, "fractional"):
            ExactDiagonalizationSolver().solve(ham)


class TestOuterLoopFailClosed(unittest.TestCase):
    def _dft(self) -> QERunResult:
        return QERunResult(run_id="dft", electronic=QEElectronicState(total_energy_ev=-1.0))

    def _quantum(self, energy: float) -> QuantumRunResult:
        return QuantumRunResult(
            energy_ev=energy,
            electronic_energy_ev=energy,
            solver_type="exact_diagonalization",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            one_rdm=[],
            operator_gradients=[],
        )

    def test_missing_required_rdm_and_gradient_do_not_pass(self):
        ledger = OuterLoopLedger(ConvergenceCriteria(require_rdm=True, require_gradient=True))
        ledger.record_iteration(self._dft(), self._quantum(-1.0))
        ledger.record_iteration(self._dft(), self._quantum(-1.0))
        result = ledger.check_convergence()
        self.assertFalse(result.is_converged)
        self.assertFalse(result.passed_criteria["rdm"])
        self.assertFalse(result.passed_criteria["gradient"])


class TestSourceResolver(unittest.TestCase):
    def test_workflow_parent_with_multiple_outputs_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.out").write_text("x")
            (root / "b.out").write_text("y")
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                resolve_qe_source_paths([str(root)])

    def test_resolver_does_not_recurse_into_serial_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "001_scf"
            child.mkdir()
            (child / "pw.out").write_text("x")
            with self.assertRaisesRegex(ValueError, "No QE"):
                resolve_qe_source_paths([str(root)])

class TestGammaModeLocatesGamma(unittest.TestCase):
    """kpoint_mode='gamma' must read the Gamma point, not whichever k-point is first."""

    @staticmethod
    def _state(coords):
        return QEElectronicState.from_energies(
            eigenvalues_ev=[[-5.0, 9.0], [-1.0, 1.0]],
            occupations=[[2.0, 0.0], [2.0, 0.0]],
            kpoint_weights=[0.5, 0.5],
            kpoint_coordinates=coords,
            n_bands=2,
            n_electrons=2.0,
            fermi_energy_ev=0.0,
        )

    def test_gamma_found_when_not_listed_first(self):
        state = self._state([(0.5, 0.5, 0.5), (0.0, 0.0, 0.0)])
        space = select_active_space(
            state, method="energy_window", emin_ev=-2.0, emax_ev=2.0, kpoint_mode="gamma"
        )
        self.assertEqual(space.active_orbitals, [0, 1])

    def test_missing_coordinates_is_an_explicit_error(self):
        state = QEElectronicState.from_energies(
            eigenvalues_ev=[[-1.0, 1.0]],
            occupations=[[2.0, 0.0]],
            kpoint_weights=[1.0],
            n_bands=2,
            n_electrons=2.0,
            fermi_energy_ev=0.0,
        )
        with self.assertRaisesRegex(ValueError, "coordinates"):
            select_active_space(state, method="energy_window", kpoint_mode="gamma")

    def test_no_gamma_in_list_is_an_explicit_error(self):
        state = self._state([(0.5, 0.5, 0.5), (0.25, 0.25, 0.25)])
        with self.assertRaisesRegex(ValueError, r"\(0, 0, 0\)"):
            select_active_space(state, method="energy_window", kpoint_mode="gamma")


class TestNonContiguousSelectionAccountsForEveryBand(unittest.TestCase):
    """Bands inside a gap are frozen, not unclassified; electrons must still add up."""

    @staticmethod
    def _state():
        return QEElectronicState.from_energies(
            eigenvalues_ev=[[-6.0, -2.0, 1.0, 4.0]],
            occupations=[[2.0, 2.0, 2.0, 2.0]],
            kpoint_weights=[1.0],
            n_bands=4,
            n_electrons=8.0,
            fermi_energy_ev=0.0,
        )

    def test_contiguous_selection_accounts_for_all_electrons(self):
        space = select_active_space(self._state(), method="band_index", band_indices=[0, 1, 2, 3])
        self.assertAlmostEqual(space.n_active_electrons + space.n_core_electrons, 8.0)

    def test_gap_bands_are_frozen_not_dropped(self):
        space = select_active_space(self._state(), method="band_index", band_indices=[0, 3])
        self.assertEqual(space.frozen_core_orbitals, [1, 2])
        self.assertAlmostEqual(space.n_active_electrons + space.n_core_electrons, 8.0)


class TestFeedbackNoOpBranchesEmitNoChange(unittest.TestCase):
    """A decision that reports no change must not rewrite the next input."""

    @staticmethod
    def _quantum(occupations):
        return QuantumRunResult(
            energy_ev=-1.0,
            electronic_energy_ev=-1.0,
            solver_type="exact_diagonalization",
            n_orbitals=2,
            n_electrons=2.0,
            n_spin_orbitals=4,
            natural_occupations=occupations,
            one_rdm=[[1.0, 0.0], [0.0, 1.0]],
        )

    def test_untriggered_occupation_policy_leaves_scf_controls_alone(self):
        decision = apply_quantum_feedback(
            QERunResult(run_id="r"), self._quantum([2.0, 0.0]), policy_name="occupation"
        )
        self.assertNotIn("ELECTRONS", decision.modified_namelists)
        self.assertNotIn("SYSTEM", decision.modified_namelists)

    def test_triggered_occupation_policy_still_adjusts_scf_controls(self):
        decision = apply_quantum_feedback(
            QERunResult(run_id="r"), self._quantum([1.2, 0.8]), policy_name="occupation"
        )
        self.assertEqual(decision.modified_namelists["ELECTRONS"]["mixing_beta"], 0.3)
        self.assertEqual(decision.modified_namelists["SYSTEM"]["smearing"], "cold")

    def test_untriggered_band_policy_emits_no_nbnd(self):
        decision = apply_quantum_feedback(
            QERunResult(run_id="r"), self._quantum([2.0, 0.0]), policy_name="active_space"
        )
        self.assertNotIn("SYSTEM", decision.modified_namelists)

    def test_unknown_band_count_does_not_invent_one(self):
        """Previously fell back to a hardcoded nbnd=8 on a run reporting no bands."""
        result = QERunResult(run_id="r")
        self.assertIsNone(result.electronic.n_bands)
        decision = apply_quantum_feedback(
            result, self._quantum([1.0, 0.9]), policy_name="active_space"
        )
        self.assertNotIn("SYSTEM", decision.modified_namelists)

    def test_triggered_band_policy_expands_from_the_known_count(self):
        result = QERunResult(run_id="r")
        result.electronic = QEElectronicState(n_bands=12)
        decision = apply_quantum_feedback(
            result, self._quantum([1.0, 0.9]), policy_name="active_space"
        )
        self.assertEqual(decision.modified_namelists["SYSTEM"]["nbnd"], 14)


if __name__ == "__main__":
    unittest.main()
