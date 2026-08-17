"""Calculation health, error, and sanity diagnostics."""

from __future__ import annotations

import math

from qeanalyzer.analysis.models import Diagnostic
from qeanalyzer.models import QERunResult


def analyze_health(result: QERunResult) -> list[Diagnostic]:
    """Perform health and sanity checks on calculation execution."""
    diagnostics: list[Diagnostic] = []

    # 1. Completion
    if result.status.completed:
        diagnostics.append(Diagnostic(
            level="OK",
            category="health",
            message="Calculation completed normally (JOB DONE)",
        ))
    else:
        diagnostics.append(Diagnostic(
            level="ERROR",
            category="health",
            message="Calculation was interrupted or terminated abnormally",
        ))

    # 2. NaN / Inf check
    tot_e = result.electronic.total_energy_ry
    if tot_e is not None and (math.isnan(tot_e) or math.isinf(tot_e)):
        diagnostics.append(Diagnostic(
            level="ERROR",
            category="health",
            message="Total energy contains NaN or Inf numeric values",
        ))
    else:
        diagnostics.append(Diagnostic(
            level="OK",
            category="health",
            message="No NaN/Inf detected in total energy",
        ))

    # 3. Warnings
    for w in result.status.warnings:
        diagnostics.append(Diagnostic(
            level="WARNING",
            category="health",
            message=f"QE Warning: {w}",
        ))

    # 4. Errors
    for e in result.status.errors:
        diagnostics.append(Diagnostic(
            level="ERROR",
            category="health",
            message=f"QE Error: {e}",
        ))

    return diagnostics
