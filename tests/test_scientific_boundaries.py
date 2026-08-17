from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qeanalyzer.io.source_bundle import resolve_qe_source_paths
from qeanalyzer.models import QEElectronicState, QERunResult
from qeanalyzer.quantum import BandIndexSelector, ExactDiagonalizationSolver
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


if __name__ == "__main__":
    unittest.main()
