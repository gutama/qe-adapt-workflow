"""SCF convergence and iteration analysis."""

from __future__ import annotations

from qeanalyzer.analysis.models import Diagnostic
from qeanalyzer.models import QERunResult


def analyze_scf(result: QERunResult) -> list[Diagnostic]:
    """Perform diagnostics on SCF convergence and iteration performance."""
    diagnostics: list[Diagnostic] = []

    # 1. Convergence
    if result.status.scf_converged:
        n_steps = result.convergence.n_scf_steps or len(result.convergence.scf_iterations)
        diagnostics.append(Diagnostic(
            level="OK",
            category="scf",
            message=f"SCF converged in {n_steps} iterations",
        ))
    else:
        diagnostics.append(Diagnostic(
            level="ERROR",
            category="scf",
            message="SCF calculation failed to converge",
            details="Consider increasing electron_maxstep, adjusting mixing_beta, or increasing conv_thr.",
        ))

    # 2. Iteration history and energy delta
    iters = result.convergence.scf_iterations
    if len(iters) >= 2:
        last_e = iters[-1].total_energy_ry
        prev_e = iters[-2].total_energy_ry
        delta_e = abs(last_e - prev_e)
        if delta_e > 1e-4 and result.status.scf_converged:
            diagnostics.append(Diagnostic(
                level="WARNING",
                category="scf",
                message=f"Large final SCF energy step: ΔE = {delta_e:.2e} Ry",
            ))

    # 3. Slow convergence warning
    if len(iters) > 50:
        diagnostics.append(Diagnostic(
            level="WARNING",
            category="scf",
            message=f"High SCF iteration count ({len(iters)} iterations)",
            details="Convergence may be struggling. Consider using a different mixing mode or smaller mixing_beta.",
        ))

    return diagnostics
