"""The shipped examples must actually run, and must not print wrong physics.

Nothing exercised examples/ before, which is how 03 came to raise "clifford_qc
excitation pool is empty" whenever the optional backend was installed -- a
failure invisible to a CI run without it.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

from qeanalyzer.quantum import clifford_qc_available

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))


def run_example(path: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=300
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{path.name} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )
    return completed.stdout


class TestExamplesRun(unittest.TestCase):
    def test_every_example_exits_cleanly(self):
        self.assertTrue(EXAMPLES, "no examples found")
        for path in EXAMPLES:
            with self.subTest(example=path.name):
                run_example(path)


class TestQuantumSolverExample(unittest.TestCase):
    """03 pairs a quantum solver with a model that has to be solvable by one."""

    PATH = Path(__file__).parent.parent / "examples" / "03_dft_to_quantum_solver.py"

    @staticmethod
    def _number(output: str, label: str) -> float:
        """Parse a printed value, scientific notation included."""
        match = re.search(
            rf"{re.escape(label)}:\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", output
        )
        assert match, f"{label!r} not found in:\n{output}"
        return float(match.group(1))

    def test_model_sector_is_not_a_closed_shell(self):
        """A filled sector gives ADAPT an empty pool and nothing to grow."""
        output = run_example(self.PATH)
        match = re.search(r"Model sector: (\d+) electrons in (\d+) orbitals", output)
        self.assertIsNotNone(match, output)
        electrons, orbitals = int(match.group(1)), int(match.group(2))
        self.assertLess(electrons, 2 * orbitals, "sector is completely filled")
        self.assertGreater(electrons, 0)

    @unittest.skipUnless(clifford_qc_available(), "clifford_qc optional sibling not installed")
    def test_adapt_reaches_the_exact_ground_state(self):
        """The band model is diagonal, so ADAPT must be run on the correlated one.

        On a diagonal model every determinant is an eigenstate: ADAPT would stop
        at the aufbau determinant with a residual gradient of 0.0 and report an
        energy above the ground state while looking converged.
        """
        output = run_example(self.PATH)
        exact = self._number(output, "Exact ground state")
        adapt = self._number(output, "clifford_qc ADAPT-VQE")
        self.assertAlmostEqual(adapt, exact, places=6)

        # On a diagonal model ADAPT stops at the reference having grown nothing.
        self.assertGreater(int(self._number(output, "Operators selected")), 0)
        self.assertLess(self._number(output, "Residual commutator gradient"), 1e-6)
