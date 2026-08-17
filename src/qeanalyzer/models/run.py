"""Canonical run result and status data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qeanalyzer.io.pw_input import PWInput
from qeanalyzer.io.pw_output import IonicStep, PWOutput, SCFIteration
from qeanalyzer.io.qe_xml import QEXMLOutput
from qeanalyzer.models.electronic import QEElectronicState
from qeanalyzer.models.provenance import QEProvenance
from qeanalyzer.models.structure import QEStructure


@dataclass
class QERunStatus:
    """Execution, convergence, and health status of a QE calculation."""

    completed: bool = False
    scf_converged: bool = False
    opt_converged: bool | None = None
    exit_status: str = "unknown"  # "converged", "not_converged", "interrupted", "error"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert status to dictionary."""
        return {
            "completed": self.completed,
            "scf_converged": self.scf_converged,
            "opt_converged": self.opt_converged,
            "exit_status": self.exit_status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QERunStatus:
        """Construct status from dictionary."""
        return cls(
            completed=data.get("completed", False),
            scf_converged=data.get("scf_converged", False),
            opt_converged=data.get("opt_converged"),
            exit_status=data.get("exit_status", "unknown"),
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
        )


@dataclass
class QEConvergenceHistory:
    """SCF iterations, optimization step history, and runtime performance."""

    n_scf_steps: int | None = None
    scf_error: float | None = None
    scf_iterations: list[SCFIteration] = field(default_factory=list)
    ionic_steps: list[IonicStep] = field(default_factory=list)
    n_bfgs_steps: int | None = None
    cpu_time_seconds: float | None = None
    wall_time_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert convergence history to dictionary."""
        return {
            "n_scf_steps": self.n_scf_steps,
            "scf_error": self.scf_error,
            "n_scf_iterations": len(self.scf_iterations),
            "scf_iterations": [
                {
                    "iteration": it.iteration,
                    "total_energy_ry": it.total_energy_ry,
                    "accuracy_ry": it.accuracy_ry,
                    "diag_method": it.diag_method,
                    "ethr": it.ethr,
                    "avg_diag_iterations": it.avg_diag_iterations,
                }
                for it in self.scf_iterations
            ],
            "n_bfgs_steps": self.n_bfgs_steps,
            "n_ionic_steps": len(self.ionic_steps),
            "timing": {
                "cpu_time_seconds": self.cpu_time_seconds,
                "wall_time_seconds": self.wall_time_seconds,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QEConvergenceHistory:
        """Construct convergence history from dictionary."""
        iters: list[SCFIteration] = []
        for it in data.get("scf_iterations", []):
            iters.append(SCFIteration(
                iteration=it["iteration"],
                total_energy_ry=it["total_energy_ry"],
                accuracy_ry=it["accuracy_ry"],
                diag_method=it.get("diag_method"),
                ethr=it.get("ethr"),
                avg_diag_iterations=it.get("avg_diag_iterations"),
            ))

        timing = data.get("timing", {})
        return cls(
            n_scf_steps=data.get("n_scf_steps"),
            scf_error=data.get("scf_error"),
            scf_iterations=iters,
            n_bfgs_steps=data.get("n_bfgs_steps"),
            cpu_time_seconds=timing.get("cpu_time_seconds"),
            wall_time_seconds=timing.get("wall_time_seconds"),
        )


@dataclass
class QERunResult:
    """Unified, canonical representation of a Quantum ESPRESSO calculation."""

    schema_version: str = "1.0"
    run_id: str | None = None
    parent_run: str | None = None
    calculation: str = "scf"
    status: QERunStatus = field(default_factory=QERunStatus)
    provenance: QEProvenance = field(default_factory=QEProvenance)
    electronic: QEElectronicState = field(default_factory=QEElectronicState)
    structure: QEStructure | None = None
    initial_structure: QEStructure | None = None
    convergence: QEConvergenceHistory = field(default_factory=QEConvergenceHistory)

    def to_dict(self) -> dict[str, Any]:
        """Convert QERunResult to structured JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "parent_run": self.parent_run,
            "calculation": self.calculation,
            "status": self.status.to_dict(),
            "provenance": self.provenance.to_dict(),
            "electronic": self.electronic.to_dict(),
            "structure": self.structure.to_dict() if self.structure else None,
            "initial_structure": (
                self.initial_structure.to_dict() if self.initial_structure else None
            ),
            "convergence": self.convergence.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QERunResult:
        """Construct QERunResult from a dictionary."""
        struct = (
            QEStructure.from_dict(data["structure"])
            if data.get("structure")
            else None
        )
        init_struct = (
            QEStructure.from_dict(data["initial_structure"])
            if data.get("initial_structure")
            else None
        )
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            run_id=data.get("run_id"),
            parent_run=data.get("parent_run"),
            calculation=data.get("calculation", "scf"),
            status=QERunStatus.from_dict(data.get("status", {})),
            provenance=QEProvenance.from_dict(data.get("provenance", {})),
            electronic=QEElectronicState.from_dict(data.get("electronic", {})),
            structure=struct,
            initial_structure=init_struct,
            convergence=QEConvergenceHistory.from_dict(data.get("convergence", {})),
        )


def build_run_result(
    pw_in: PWInput | None = None,
    pw_out: PWOutput | None = None,
    qe_xml: QEXMLOutput | None = None,
    run_id: str | None = None,
    parent_run: str | None = None,
    input_text: str | None = None,
) -> QERunResult:
    """Build a unified canonical QERunResult combining all available data sources.

    XML data is treated as ground truth for numerical energies, eigenvalues,
    and cell coordinates. pw_out provides diagnostics, convergence step
    histories, timing, and error information. pw_in provides input provenance.

    Parameters
    ----------
    pw_in : PWInput, optional
        Parsed pw.x input.
    pw_out : PWOutput, optional
        Parsed stdout output.
    qe_xml : QEXMLOutput, optional
        Parsed structured XML.
    run_id : str, optional
        Unique ID for this run.
    parent_run : str, optional
        ID of parent run in workflow DAG.
    input_text : str, optional
        Raw text of pw.in for SHA256 hashing.

    Returns
    -------
    QERunResult
        Consolidated run result.
    """
    # 1. Calculation type
    calc = "scf"
    if qe_xml and qe_xml.calculation:
        calc = qe_xml.calculation
    elif pw_out and pw_out.calculation:
        calc = pw_out.calculation
    elif pw_in:
        calc = pw_in.calculation

    # 2. Provenance
    input_hash = None
    if input_text:
        input_hash = QEProvenance.compute_sha256(input_text)

    qe_ver = None
    if qe_xml and qe_xml.creator_version:
        qe_ver = qe_xml.creator_version
    elif pw_out and pw_out.qe_version:
        qe_ver = pw_out.qe_version

    n_mpi = (
        qe_xml.nprocs
        if qe_xml and qe_xml.nprocs is not None
        else (pw_out.n_mpi_processes if pw_out else None)
    )
    n_th = (
        qe_xml.nthreads
        if qe_xml and qe_xml.nthreads is not None
        else (pw_out.n_threads if pw_out else None)
    )
    prefix = (
        qe_xml.prefix
        if qe_xml and qe_xml.prefix
        else (pw_in.namelists.get("CONTROL", {}).get("prefix") if pw_in else None)
    )
    outdir = pw_in.namelists.get("CONTROL", {}).get("outdir") if pw_in else None

    provenance = QEProvenance(
        run_id=run_id,
        parent_run=parent_run,
        qe_version=qe_ver,
        input_sha256=input_hash,
        prefix=prefix,
        outdir=outdir,
        n_mpi_processes=n_mpi,
        n_threads=n_th,
    )

    # 3. Status
    completed = False
    scf_conv = False
    opt_conv = None
    exit_st = "unknown"
    warnings_list: list[str] = []
    errors_list: list[str] = []

    if pw_out:
        completed = pw_out.job_done
        scf_conv = pw_out.scf_converged
        opt_conv = pw_out.bfgs_converged if pw_out.bfgs_converged else None
        exit_st = pw_out.exit_status or "unknown"
        warnings_list.extend(pw_out.warnings)
        errors_list.extend(pw_out.errors)

    if qe_xml:
        if qe_xml.scf_converged:
            scf_conv = True
        if qe_xml.opt_converged is not None:
            opt_conv = qe_xml.opt_converged
        if not pw_out:
            completed = True
            exit_st = "converged" if scf_conv else "not_converged"

    status = QERunStatus(
        completed=completed,
        scf_converged=scf_conv,
        opt_converged=opt_conv,
        exit_status=exit_st,
        warnings=warnings_list,
        errors=errors_list,
    )

    # 4. Electronic state
    tot_ry = None
    tot_ha = None
    one_e_ry = None
    hart_ry = None
    xc_ry = None
    ewald_ry = None
    smear_ry = None
    fermi_ev = None
    homo_ev = None
    lumo_ev = None
    nelec = None
    nbnd = None
    nk = None
    lsda = False
    noncolin = False
    spinorbit = False
    mag_tot = None
    mag_abs = None
    eigs_ev: list[list[float]] = []
    occs: list[list[float]] = []

    if qe_xml:
        tot_ha = qe_xml.total_energy_hartree
        tot_ry = qe_xml.total_energy_ry
        one_e_ry = (
            qe_xml.one_electron_hartree * 2.0
            if qe_xml.one_electron_hartree is not None
            else None
        )
        hart_ry = (
            qe_xml.hartree_energy_hartree * 2.0
            if qe_xml.hartree_energy_hartree is not None
            else None
        )
        xc_ry = (
            qe_xml.xc_energy_hartree * 2.0
            if qe_xml.xc_energy_hartree is not None
            else None
        )
        ewald_ry = (
            qe_xml.ewald_energy_hartree * 2.0
            if qe_xml.ewald_energy_hartree is not None
            else None
        )
        smear_ry = (
            qe_xml.demet_energy_hartree * 2.0
            if qe_xml.demet_energy_hartree is not None
            else None
        )
        fermi_ev = qe_xml.fermi_energy_ev
        homo_ev = qe_xml.highest_occupied_ev
        lumo_ev = qe_xml.lowest_unoccupied_ev
        nelec = qe_xml.n_electrons
        nbnd = qe_xml.n_bands
        nk = qe_xml.n_kpoints
        lsda = qe_xml.lsda
        noncolin = qe_xml.noncolin
        spinorbit = qe_xml.spinorbit
        eigs_ev = [kp.eigenvalues_ev for kp in qe_xml.kpoints]
        occs = [kp.occupations for kp in qe_xml.kpoints]

    if pw_out:
        if tot_ry is None:
            tot_ry = pw_out.total_energy_ry
        if one_e_ry is None:
            one_e_ry = pw_out.one_electron_ry
        if hart_ry is None:
            hart_ry = pw_out.hartree_ry
        if xc_ry is None:
            xc_ry = pw_out.xc_ry
        if ewald_ry is None:
            ewald_ry = pw_out.ewald_ry
        if smear_ry is None:
            smear_ry = pw_out.smearing_ry
        if fermi_ev is None:
            fermi_ev = pw_out.fermi_energy_ev
        if homo_ev is None:
            homo_ev = pw_out.highest_occupied_ev
        if lumo_ev is None:
            lumo_ev = pw_out.lowest_unoccupied_ev
        if nelec is None:
            nelec = pw_out.n_electrons
        if nbnd is None:
            nbnd = pw_out.n_bands
        if nk is None:
            nk = pw_out.n_kpoints
        mag_tot = pw_out.total_magnetization
        mag_abs = pw_out.absolute_magnetization

    if pw_in:
        if nelec is None and pw_in.nat is not None:
            pass  # nelec from input requires atomic species valence

    electronic = QEElectronicState.from_energies(
        total_energy_ry=tot_ry,
        total_energy_hartree=tot_ha,
        one_electron_ry=one_e_ry,
        hartree_energy_ry=hart_ry,
        xc_energy_ry=xc_ry,
        ewald_energy_ry=ewald_ry,
        smearing_energy_ry=smear_ry,
        fermi_energy_ev=fermi_ev,
        highest_occupied_ev=homo_ev,
        lowest_unoccupied_ev=lumo_ev,
        n_electrons=nelec,
        n_bands=nbnd,
        n_kpoints=nk,
        total_magnetization=mag_tot,
        absolute_magnetization=mag_abs,
        eigenvalues_ev=eigs_ev,
        occupations=occs,
        lsda=lsda,
        noncolin=noncolin,
        spinorbit=spinorbit,
    )

    # 5. Structure (Initial and Final/Current)
    structure = None
    init_structure = None

    if qe_xml:
        if qe_xml.initial_structure:
            init_structure = QEStructure.from_cell_and_atoms_bohr(
                cell_bohr=qe_xml.initial_structure.cell_vectors_bohr,
                atoms_bohr=qe_xml.initial_structure.positions_bohr,
                alat_bohr=qe_xml.initial_structure.alat_bohr,
            )

        active_s = qe_xml.final_structure or qe_xml.initial_structure
        if active_s:
            structure = QEStructure.from_cell_and_atoms_bohr(
                cell_bohr=active_s.cell_vectors_bohr,
                atoms_bohr=active_s.positions_bohr,
                alat_bohr=active_s.alat_bohr,
                forces_ry_bohr=qe_xml.forces_ry_bohr,
                stress_kbar=qe_xml.stress_kbar,
                pressure_kbar=qe_xml.pressure_kbar,
            )

    if structure is None and pw_out and pw_out.forces:
        # Construct structure from pw_out forces and pw_in if available
        pass

    # 6. Convergence history & timing
    scf_steps = None
    scf_err = None
    scf_iters: list[SCFIteration] = []
    ionic_steps: list[IonicStep] = []
    n_bfgs = None
    cpu_s = None
    wall_s = None

    if qe_xml:
        scf_steps = qe_xml.n_scf_steps
        scf_err = qe_xml.scf_error

    if pw_out:
        if scf_steps is None:
            scf_steps = pw_out.n_scf_iterations
        scf_iters = list(pw_out.scf_iterations)
        ionic_steps = list(pw_out.ionic_steps)
        n_bfgs = pw_out.n_bfgs_steps
        cpu_s = pw_out.cpu_time_seconds
        wall_s = pw_out.wall_time_seconds

    convergence = QEConvergenceHistory(
        n_scf_steps=scf_steps,
        scf_error=scf_err,
        scf_iterations=scf_iters,
        ionic_steps=ionic_steps,
        n_bfgs_steps=n_bfgs,
        cpu_time_seconds=cpu_s,
        wall_time_seconds=wall_s,
    )

    return QERunResult(
        run_id=run_id,
        parent_run=parent_run,
        calculation=calc,
        status=status,
        provenance=provenance,
        electronic=electronic,
        structure=structure,
        initial_structure=init_structure,
        convergence=convergence,
    )
