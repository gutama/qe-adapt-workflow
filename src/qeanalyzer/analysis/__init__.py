"""Analysis and diagnostic evaluation routines."""

from qeanalyzer.analysis.electronic import analyze_electronic
from qeanalyzer.analysis.health import analyze_health
from qeanalyzer.analysis.models import Diagnostic, DiagnosticLevel
from qeanalyzer.analysis.scf import analyze_scf
from qeanalyzer.analysis.structure import analyze_structure
from qeanalyzer.models import QERunResult


def run_all_diagnostics(result: QERunResult) -> list[Diagnostic]:
    """Execute all diagnostic checks across SCF, electronic, structure, and health categories.

    Parameters
    ----------
    result : QERunResult
        The calculation result to analyze.

    Returns
    -------
    list[Diagnostic]
        Consolidated list of diagnostics.
    """
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(analyze_health(result))
    diagnostics.extend(analyze_scf(result))
    diagnostics.extend(analyze_electronic(result))
    diagnostics.extend(analyze_structure(result))
    return diagnostics


__all__ = [
    "Diagnostic",
    "DiagnosticLevel",
    "analyze_electronic",
    "analyze_health",
    "analyze_scf",
    "analyze_structure",
    "run_all_diagnostics",
]
