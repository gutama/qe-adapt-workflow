"""Structural geometry, force, and stress diagnostics."""

from __future__ import annotations

from qeanalyzer.analysis.models import Diagnostic
from qeanalyzer.models import QERunResult


def analyze_structure(result: QERunResult) -> list[Diagnostic]:
    """Perform diagnostics on structure, forces, and stress."""
    diagnostics: list[Diagnostic] = []
    struct = result.structure

    if struct is not None:
        if struct.max_force_ry_bohr is not None:
            if struct.max_force_ry_bohr > 0.05:
                diagnostics.append(Diagnostic(
                    level="WARNING",
                    category="structure",
                    message=f"Large residual forces on atoms: max force = {struct.max_force_ry_bohr:.4f} Ry/Bohr",
                    details="Geometry may need relaxation (calculation = 'relax' or 'vc-relax').",
                ))
            else:
                diagnostics.append(Diagnostic(
                    level="OK",
                    category="structure",
                    message=f"Atomic forces within threshold: max force = {struct.max_force_ry_bohr:.4f} Ry/Bohr",
                ))

        if struct.pressure_kbar is not None:
            if abs(struct.pressure_kbar) > 10.0 and result.calculation == "vc-relax":
                diagnostics.append(Diagnostic(
                    level="WARNING",
                    category="structure",
                    message=f"Non-negligible cell pressure: P = {struct.pressure_kbar:.2f} kbar",
                ))

    return diagnostics
