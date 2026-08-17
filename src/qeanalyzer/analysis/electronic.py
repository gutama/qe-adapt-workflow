"""Electronic structure, band gap, and occupation diagnostics."""

from __future__ import annotations

from qeanalyzer.analysis.models import Diagnostic
from qeanalyzer.models import QERunResult


def analyze_electronic(result: QERunResult) -> list[Diagnostic]:
    """Perform diagnostics on electronic state and occupations."""
    diagnostics: list[Diagnostic] = []
    elec = result.electronic

    if elec.is_metal:
        fermi = f"{elec.fermi_energy_ev:.3f} eV" if elec.fermi_energy_ev is not None else "N/A"
        diagnostics.append(Diagnostic(
            level="INFO",
            category="electronic",
            message=f"Metallic system (Fermi level: {fermi})",
        ))
    elif elec.band_gap_ev is not None:
        diagnostics.append(Diagnostic(
            level="INFO",
            category="electronic",
            message=f"Insulator / semiconductor with band gap: {elec.band_gap_ev:.3f} eV",
        ))

    # Check top band occupations
    if elec.occupations:
        for ik, occ_list in enumerate(elec.occupations):
            if occ_list and occ_list[-1] > 1e-3 and not elec.is_metal:
                diagnostics.append(Diagnostic(
                    level="WARNING",
                    category="electronic",
                    message=f"Highest band has non-zero occupation at k-point #{ik+1} (occ = {occ_list[-1]:.3f})",
                    details="The number of bands (nbnd) may be too small and truncating occupied states.",
                ))
                break

    return diagnostics
