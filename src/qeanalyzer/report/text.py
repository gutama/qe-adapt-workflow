"""Human-readable text and Markdown report generator."""

from __future__ import annotations

from pathlib import Path

from qeanalyzer.analysis import Diagnostic, run_all_diagnostics
from qeanalyzer.models import QERunResult


def generate_text_report(
    result: QERunResult,
    diagnostics: list[Diagnostic] | None = None,
    markdown: bool = False,
) -> str:
    """Generate a structured, human-readable summary report of a QE run.

    Parameters
    ----------
    result : QERunResult
        The calculation result to report.
    diagnostics : list[Diagnostic], optional
        Pre-computed diagnostics (computed automatically if None).
    markdown : bool, optional
        If True, format with GitHub-flavored Markdown headings and badges.

    Returns
    -------
    str
        Formatted text or Markdown report.
    """
    if diagnostics is None:
        diagnostics = run_all_diagnostics(result)

    lines: list[str] = []

    def section(title: str) -> None:
        if markdown:
            lines.append(f"\n## {title}\n")
        else:
            lines.append(f"\n{title}")
            lines.append("-" * len(title))

    # Title
    if markdown:
        lines.append("# Quantum ESPRESSO Run Analysis")
    else:
        lines.append("Quantum ESPRESSO Run Analysis")
        lines.append("=" * 29)

    # 1. Run Metadata
    section("Run Metadata")
    if result.run_id:
        lines.append(f"{'Run ID':<20}: {result.run_id}")
    if result.parent_run:
        lines.append(f"{'Parent Run':<20}: {result.parent_run}")
    lines.append(f"{'Calculation':<20}: {result.calculation}")
    lines.append(f"{'Status':<20}: {result.status.exit_status.upper()}")
    if result.provenance.qe_version:
        lines.append(f"{'QE version':<20}: {result.provenance.qe_version}")
    if result.provenance.input_sha256:
        lines.append(f"{'Input SHA-256':<20}: {result.provenance.input_sha256[:12]}...")

    # 2. Electronic Structure
    section("Electronic Structure")
    elec = result.electronic
    if elec.total_energy_ry is not None:
        ev_str = f" ({elec.total_energy_ev:.4f} eV)" if elec.total_energy_ev is not None else ""
        lines.append(f"{'Total energy':<20}: {elec.total_energy_ry:.8f} Ry{ev_str}")
    if elec.one_electron_ry is not None:
        lines.append(f"{'One-electron energy':<20}: {elec.one_electron_ry:.8f} Ry")
    if elec.hartree_energy_ry is not None:
        lines.append(f"{'Hartree energy':<20}: {elec.hartree_energy_ry:.8f} Ry")
    if elec.xc_energy_ry is not None:
        lines.append(f"{'XC energy':<20}: {elec.xc_energy_ry:.8f} Ry")
    if elec.ewald_energy_ry is not None:
        lines.append(f"{'Ewald energy':<20}: {elec.ewald_energy_ry:.8f} Ry")
    if elec.smearing_energy_ry is not None:
        lines.append(f"{'Smearing energy':<20}: {elec.smearing_energy_ry:.8f} Ry")

    if elec.fermi_energy_ev is not None:
        lines.append(f"{'Fermi energy':<20}: {elec.fermi_energy_ev:.4f} eV")
    if elec.highest_occupied_ev is not None:
        lines.append(f"{'Highest occupied':<20}: {elec.highest_occupied_ev:.4f} eV")
    if elec.lowest_unoccupied_ev is not None:
        lines.append(f"{'Lowest unoccupied':<20}: {elec.lowest_unoccupied_ev:.4f} eV")
    if elec.band_gap_ev is not None:
        lines.append(f"{'Band gap':<20}: {elec.band_gap_ev:.4f} eV")

    if elec.n_electrons is not None:
        lines.append(f"{'Number of electrons':<20}: {elec.n_electrons}")
    if elec.n_bands is not None:
        lines.append(f"{'Number of bands':<20}: {elec.n_bands}")
    if elec.n_kpoints is not None:
        lines.append(f"{'Number of k-points':<20}: {elec.n_kpoints}")

    # 3. Structure & Forces
    if result.structure is not None:
        section("Structure & Forces")
        s = result.structure
        if s.alat_bohr is not None:
            ang_str = f" ({s.alat_angstrom:.4f} Å)" if s.alat_angstrom is not None else ""
            lines.append(f"{'Lattice parameter':<20}: {s.alat_bohr:.4f} Bohr{ang_str}")
        if s.volume_bohr3 is not None:
            ang3_str = f" ({s.volume_angstrom3:.3f} Å³)" if s.volume_angstrom3 is not None else ""
            lines.append(f"{'Unit-cell volume':<20}: {s.volume_bohr3:.3f} Bohr³{ang3_str}")
        lines.append(f"{'Number of atoms':<20}: {len(s.atoms)}")
        if s.max_force_ry_bohr is not None:
            lines.append(f"{'Max atomic force':<20}: {s.max_force_ry_bohr:.6f} Ry/Bohr")
        if s.total_force_ry_bohr is not None:
            lines.append(f"{'Total force':<20}: {s.total_force_ry_bohr:.6f} Ry/Bohr")
        if s.pressure_kbar is not None:
            lines.append(f"{'Cell pressure':<20}: {s.pressure_kbar:.2f} kbar")

    # 4. Convergence & Performance
    section("Convergence & Performance")
    conv = result.convergence
    if conv.n_scf_steps is not None:
        lines.append(f"{'SCF iterations':<20}: {conv.n_scf_steps}")
    elif conv.scf_iterations:
        lines.append(f"{'SCF iterations':<20}: {len(conv.scf_iterations)}")

    if conv.scf_error is not None:
        lines.append(f"{'Final SCF error':<20}: {conv.scf_error:.2e} Ry")

    if conv.n_bfgs_steps is not None:
        lines.append(f"{'BFGS ionic steps':<20}: {conv.n_bfgs_steps}")
    elif conv.ionic_steps:
        lines.append(f"{'Ionic steps':<20}: {len(conv.ionic_steps)}")

    if conv.cpu_time_seconds is not None:
        lines.append(f"{'CPU time':<20}: {conv.cpu_time_seconds:.2f} s")
    if conv.wall_time_seconds is not None:
        lines.append(f"{'Wall time':<20}: {conv.wall_time_seconds:.2f} s")

    # 5. Diagnostics
    section("Diagnostics")
    for diag in diagnostics:
        symbol = diag.symbol
        if markdown:
            lines.append(f"- **{symbol}** [{diag.category.upper()}] {diag.message}")
        else:
            lines.append(f"{symbol} [{diag.category.upper()}] {diag.message}")
        if diag.details:
            prefix = "    " if not markdown else "  > "
            lines.append(f"{prefix}{diag.details}")

    # 6. Recommended Next Run
    section("Recommended Next Step")
    if result.status.scf_converged:
        if result.calculation == "scf":
            lines.append("NSCF calculation (scf -> nscf) for DOS/Bands, or Active-Space selection for ADAPT-VQE.")
        elif result.calculation in ("relax", "vc-relax"):
            lines.append("SCF calculation (relax -> scf) on final converged structure.")
        elif result.calculation == "nscf":
            lines.append("Bands or DOS post-processing, or Wannier90 Hamiltonian construction.")
        else:
            lines.append("Continue serial workflow chain.")
    else:
        lines.append("Resolve SCF/geometry convergence issues before progressing to downstream calculations.")

    return "\n".join(lines) + "\n"


def save_text_report(
    result: QERunResult,
    path: str | Path,
    markdown: bool = False,
) -> None:
    """Save a human-readable text report to a file on disk.

    Parameters
    ----------
    result : QERunResult
        Calculation result.
    path : str or Path
        Target destination file.
    markdown : bool, optional
        Whether to generate Markdown format.
    """
    content = generate_text_report(result, markdown=markdown)
    Path(path).write_text(content, encoding="utf-8")
