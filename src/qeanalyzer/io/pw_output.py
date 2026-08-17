"""Parse Quantum ESPRESSO pw.x text output files.

Extracts structured data from pw.x stdout: header information,
SCF iteration histories, energy components, electronic structure,
forces, stress, ionic steps, timing, and run status.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# -- Data classes ------------------------------------------------------------

@dataclass
class SCFIteration:
    """A single SCF iteration record."""

    iteration: int
    total_energy_ry: float
    accuracy_ry: float
    diag_method: str | None = None  # "Davidson" or "CG"
    ethr: float | None = None
    avg_diag_iterations: float | None = None


@dataclass
class ForceAtom:
    """Force on a single atom."""

    atom_index: int  # 1-based
    symbol: str  # atom type label (resolved from type index)
    force: tuple[float, float, float]  # Ry/au


@dataclass
class StressTensor:
    """Stress tensor and scalar pressure."""

    tensor_kbar: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    pressure_kbar: float


@dataclass
class IonicStep:
    """One ionic (BFGS) optimization step in relax/vc-relax."""

    step: int
    scf_iterations: list[SCFIteration] = field(default_factory=list)
    converged: bool = False
    total_energy_ry: float | None = None
    forces: list[ForceAtom] | None = None
    total_force: float | None = None
    stress: StressTensor | None = None


@dataclass
class PWOutput:
    """Complete parsed representation of a pw.x text output file."""

    # Header
    qe_version: str | None = None
    n_mpi_processes: int | None = None
    n_threads: int | None = None
    calculation: str | None = None

    # System
    alat_bohr: float | None = None
    cell_volume_bohr3: float | None = None
    nat: int | None = None
    ntyp: int | None = None
    n_electrons: float | None = None
    n_bands: int | None = None
    ecutwfc_ry: float | None = None
    ecutrho_ry: float | None = None
    n_kpoints: int | None = None
    mixing_beta: float | None = None

    # SCF history (for single-point calcs this is the complete set;
    # for relax/vc-relax this is the LAST electronic minimization)
    scf_iterations: list[SCFIteration] = field(default_factory=list)
    scf_converged: bool = False
    n_scf_iterations: int | None = None

    # Final energy components
    total_energy_ry: float | None = None
    one_electron_ry: float | None = None
    hartree_ry: float | None = None
    xc_ry: float | None = None
    ewald_ry: float | None = None
    smearing_ry: float | None = None

    # Electronic structure
    fermi_energy_ev: float | None = None
    highest_occupied_ev: float | None = None
    lowest_unoccupied_ev: float | None = None
    total_magnetization: float | None = None
    absolute_magnetization: float | None = None

    # Forces (final)
    forces: list[ForceAtom] | None = None
    total_force: float | None = None
    scf_correction_force: float | None = None
    stress: StressTensor | None = None

    # Ionic steps (relax / vc-relax only)
    ionic_steps: list[IonicStep] = field(default_factory=list)
    n_bfgs_steps: int | None = None
    bfgs_converged: bool = False

    # Timing
    wall_time_seconds: float | None = None
    cpu_time_seconds: float | None = None

    # Run status
    job_done: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    exit_status: str | None = None


# -- Helpers -----------------------------------------------------------------

_RE_TIME_COMPONENT = re.compile(
    r"(?:(\d+(?:\.\d+)?)d)?"
    r"(?:(\d+(?:\.\d+)?)h)?"
    r"(?:(\d+(?:\.\d+)?)m)?"
    r"(?:(\d+(?:\.\d+)?)s)?"
)


def _parse_time(time_str: str) -> float | None:
    """Convert QE timing string (e.g., '0.42s', '1m 2.34s', '1h 2m') to seconds."""
    s = time_str.strip().replace(" ", "")
    if not s:
        return None

    # Direct float with 's' at the end (e.g., 0.42s)
    if s.endswith("s") and s[:-1].replace(".", "", 1).isdigit():
        return float(s[:-1])

    m = _RE_TIME_COMPONENT.fullmatch(s)
    if not m:
        return None

    days_s, hours_s, mins_s, secs_s = m.groups()
    total = 0.0
    if days_s:
        total += float(days_s) * 86400.0
    if hours_s:
        total += float(hours_s) * 3600.0
    if mins_s:
        total += float(mins_s) * 60.0
    if secs_s:
        total += float(secs_s)
    return total


# -- Regex patterns ----------------------------------------------------------

_RE_VERSION = re.compile(r"Program PWSCF v\.(\S+)")
_RE_PARALLEL = re.compile(r"running on\s+(\d+)\s+processor", re.IGNORECASE)
_RE_THREADS = re.compile(r"Threads/MPI process:\s+(\d+)", re.IGNORECASE)

_RE_ALAT = re.compile(r"lattice parameter \(BOHR\)\s*=\s*([\d.]+)", re.IGNORECASE)
_RE_VOLUME = re.compile(r"unit-cell volume\s*=\s*([\d.]+)", re.IGNORECASE)
_RE_NAT = re.compile(r"number of atoms/cell\s*=\s*(\d+)", re.IGNORECASE)
_RE_NTYP = re.compile(r"number of atomic types\s*=\s*(\d+)", re.IGNORECASE)
_RE_NELEC = re.compile(r"number of electrons\s*=\s*([\d.]+)", re.IGNORECASE)
_RE_NBND = re.compile(r"number of Kohn-Sham states\s*=\s*(\d+)", re.IGNORECASE)
_RE_ECUTWFC = re.compile(r"kinetic-energy cutoff\s*=\s*([\d.]+)\s*Ry", re.IGNORECASE)
_RE_ECUTRHO = re.compile(r"charge-density cutoff\s*=\s*([\d.]+)\s*Ry", re.IGNORECASE)
_RE_NKPTS = re.compile(r"number of k points\s*=\s*(\d+)", re.IGNORECASE)
_RE_MIXING = re.compile(r"mixing beta\s*=\s*([\d.]+)", re.IGNORECASE)
_RE_CALCULATION = re.compile(r"calculation\s*=\s*['\"]?(\w+)['\"]?", re.IGNORECASE)

# Species table
_RE_SPECIES_LINE = re.compile(r"^\s+([A-Za-z][A-Za-z0-9]*)\s+([\d.]+)\s+([\d.]+)\s+(\S+)")

# SCF Iteration
_RE_ITERATION = re.compile(
    r"iteration\s*#\s*(\d+)\s+ecut=\s*([\d.]+)\s*Ry\s+beta=\s*([\d.]+)",
    re.IGNORECASE,
)
_RE_DIAG_METHOD = re.compile(
    r"(Davidson diagonalization with overlap|CG style diagonalization|PPCG style diagonalization)",
    re.IGNORECASE,
)
_RE_ETHR = re.compile(
    r"ethr =\s*([\d.E+-]+),\s*avg # of iterations\s*=\s*([\d.]+)",
    re.IGNORECASE,
)
_RE_TOTAL_ENERGY = re.compile(
    r"total energy\s*=\s*([-\d.]+)\s*Ry",
    re.IGNORECASE,
)
_RE_ACCURACY = re.compile(
    r"estimated scf accuracy\s*<\s*([\d.Ee+-]+)\s*Ry",
    re.IGNORECASE,
)

# Convergence
_RE_CONVERGED = re.compile(
    r"convergence has been achieved in\s+(\d+)\s+iterations",
    re.IGNORECASE,
)
_RE_NOT_CONVERGED = re.compile(
    r"convergence NOT achieved",
    re.IGNORECASE,
)

# Final energy components
_RE_FINAL_ENERGY = re.compile(
    r"^!\s+total energy\s*=\s*([-\d.]+)\s*Ry",
    re.IGNORECASE,
)
_RE_ONE_ELECTRON = re.compile(
    r"one-electron contribution\s*=\s*([-\d.]+)\s*Ry",
    re.IGNORECASE,
)
_RE_HARTREE = re.compile(
    r"hartree contribution\s*=\s*([-\d.]+)\s*Ry",
    re.IGNORECASE,
)
_RE_XC = re.compile(
    r"xc contribution\s*=\s*([-\d.]+)\s*Ry",
    re.IGNORECASE,
)
_RE_EWALD = re.compile(
    r"ewald contribution\s*=\s*([-\d.]+)\s*Ry",
    re.IGNORECASE,
)
_RE_SMEARING = re.compile(
    r"smearing contrib\.\s*\(-TS\)\s*=\s*([-\d.]+)\s*Ry",
    re.IGNORECASE,
)

# Electronic
_RE_FERMI = re.compile(
    r"the Fermi energy is\s+([-\d.]+)\s*ev",
    re.IGNORECASE,
)
_RE_BAND_EDGES = re.compile(
    r"highest occupied, lowest unoccupied level \(ev\):\s+([-\d.]+)\s+([-\d.]+)",
    re.IGNORECASE,
)
_RE_TOTAL_MAG = re.compile(
    r"total magnetization\s*=\s*([-\d.]+)\s*Bohr mag/cell",
    re.IGNORECASE,
)
_RE_ABS_MAG = re.compile(
    r"absolute magnetization\s*=\s*([\d.]+)\s*Bohr mag/cell",
    re.IGNORECASE,
)

# Forces
_RE_FORCE_ATOM = re.compile(
    r"atom\s+(\d+)\s+type\s+(\d+)\s+force\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
    re.IGNORECASE,
)
_RE_TOTAL_FORCE = re.compile(
    r"Total force\s*=\s*([\d.]+)\s+Total SCF correction\s*=\s*([\d.]+)",
    re.IGNORECASE,
)

# Stress
_RE_STRESS_HEADER = re.compile(
    r"total\s+stress.*P=\s*([-\d.]+)",
    re.IGNORECASE,
)

# Timing
_RE_TIMING = re.compile(
    r"PWSCF\s*:\s*(.+?)\s+CPU\s+(.+?)\s+WALL",
    re.IGNORECASE,
)

# BFGS
_RE_BFGS_CONVERGED = re.compile(
    r"bfgs converged in\s+(\d+)\s+scf cycles and\s+(\d+)\s+bfgs steps",
    re.IGNORECASE,
)
_RE_N_SCF_CYCLES = re.compile(
    r"number of scf cycles\s*=\s*(\d+)",
    re.IGNORECASE,
)
_RE_N_BFGS_STEPS = re.compile(
    r"number of bfgs steps\s*=\s*(\d+)",
    re.IGNORECASE,
)

# JOB DONE / Errors / Warnings
_RE_JOB_DONE = re.compile(r"JOB DONE", re.IGNORECASE)
_RE_WARNING = re.compile(r"^\s*(?:%\s*)?Warning:\s*(.+)", re.IGNORECASE)
_RE_ERROR = re.compile(r"^\s*(?:%\s*)?Error in routine\s+(.+)", re.IGNORECASE)


# -- Main Parser -------------------------------------------------------------

def parse_pw_output(text: str) -> PWOutput:
    """Parse a complete pw.x text output file.

    Parameters
    ----------
    text : str
        Full text content of the stdout file.

    Returns
    -------
    PWOutput
        Structured representation of the run output.
    """
    output = PWOutput()
    lines = text.replace("\r\n", "\n").split("\n")

    species_map: dict[int, str] = {}  # 1-based type index -> symbol
    in_species_table = False

    # Track SCF iterations & ionic steps
    all_ionic_steps: list[IonicStep] = []
    current_step_scf: list[SCFIteration] = []
    # Electronic convergence of the ionic step currently being accumulated. Reset
    # at each new ionic block so a step that failed to converge is recorded as
    # such instead of inheriting the previous step's result.
    current_step_converged: bool = False
    current_iter_num: int | None = None
    current_iter_energy: float | None = None
    current_iter_acc: float | None = None
    current_iter_diag: str | None = None
    current_iter_ethr: float | None = None
    current_iter_avg_diag: float | None = None

    current_forces: list[ForceAtom] | None = None
    current_total_force: float | None = None
    current_scf_corr: float | None = None
    current_stress: StressTensor | None = None
    current_final_energy: float | None = None

    reading_forces = False
    reading_stress = False
    stress_lines_remaining = 0
    stress_rows_kbar: list[tuple[float, float, float]] = []
    stress_pressure: float = 0.0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # 1. Header
        if m := _RE_VERSION.search(line):
            output.qe_version = m.group(1)
        elif m := _RE_PARALLEL.search(line):
            output.n_mpi_processes = int(m.group(1))
        elif m := _RE_THREADS.search(line):
            output.n_threads = int(m.group(1))
        elif m := _RE_CALCULATION.search(line):
            if output.calculation is None:
                output.calculation = m.group(1).lower()

        # 2. System info
        elif m := _RE_ALAT.search(line):
            output.alat_bohr = float(m.group(1))
        elif m := _RE_VOLUME.search(line):
            output.cell_volume_bohr3 = float(m.group(1))
        elif m := _RE_NAT.search(line):
            output.nat = int(m.group(1))
        elif m := _RE_NTYP.search(line):
            output.ntyp = int(m.group(1))
        elif m := _RE_NELEC.search(line):
            output.n_electrons = float(m.group(1))
        elif m := _RE_NBND.search(line):
            output.n_bands = int(m.group(1))
        elif m := _RE_ECUTWFC.search(line):
            output.ecutwfc_ry = float(m.group(1))
        elif m := _RE_ECUTRHO.search(line):
            output.ecutrho_ry = float(m.group(1))
        elif m := _RE_NKPTS.search(line):
            output.n_kpoints = int(m.group(1))
        elif m := _RE_MIXING.search(line):
            output.mixing_beta = float(m.group(1))

        # 3. Species table
        elif "atomic species" in line and "valence" in line:
            in_species_table = True
        elif in_species_table:
            if m := _RE_SPECIES_LINE.match(line):
                sym = m.group(1)
                idx = len(species_map) + 1
                species_map[idx] = sym
            else:
                in_species_table = False

        # 4. New ionic calculation block detection
        elif "new calculation: restart with different structure" in line or "Self-consistent Calculation" in line:
            # If we already have accumulated SCF iterations and forces, archive this step
            if current_step_scf and current_forces is not None:
                all_ionic_steps.append(IonicStep(
                    step=len(all_ionic_steps) + 1,
                    scf_iterations=list(current_step_scf),
                    converged=current_step_converged,
                    total_energy_ry=current_final_energy,
                    forces=current_forces,
                    total_force=current_total_force,
                    stress=current_stress,
                ))
                current_step_scf.clear()
                current_step_converged = False
                current_forces = None
                current_total_force = None
                current_stress = None
                current_final_energy = None

        # 5. SCF Iteration parsing
        elif m := _RE_ITERATION.search(line):
            # If there was a previous uncommitted iteration, commit it
            if current_iter_num is not None and current_iter_energy is not None and current_iter_acc is not None:
                current_step_scf.append(SCFIteration(
                    iteration=current_iter_num,
                    total_energy_ry=current_iter_energy,
                    accuracy_ry=current_iter_acc,
                    diag_method=current_iter_diag,
                    ethr=current_iter_ethr,
                    avg_diag_iterations=current_iter_avg_diag,
                ))
            current_iter_num = int(m.group(1))
            current_iter_energy = None
            current_iter_acc = None
            current_iter_diag = None
            current_iter_ethr = None
            current_iter_avg_diag = None

        elif m := _RE_DIAG_METHOD.search(line):
            diag_raw = m.group(1).lower()
            if "davidson" in diag_raw:
                current_iter_diag = "Davidson"
            elif "cg" in diag_raw:
                current_iter_diag = "CG"
            else:
                current_iter_diag = m.group(1)

        elif m := _RE_ETHR.search(line):
            current_iter_ethr = float(m.group(1).replace("D", "E").replace("d", "e"))
            current_iter_avg_diag = float(m.group(2))

        elif not line.strip().startswith("!") and (m := _RE_TOTAL_ENERGY.search(line)):
            current_iter_energy = float(m.group(1))

        elif m := _RE_ACCURACY.search(line):
            current_iter_acc = float(m.group(1).replace("D", "E").replace("d", "e"))
            # Commit the iteration record immediately
            if current_iter_num is not None and current_iter_energy is not None:
                current_step_scf.append(SCFIteration(
                    iteration=current_iter_num,
                    total_energy_ry=current_iter_energy,
                    accuracy_ry=current_iter_acc,
                    diag_method=current_iter_diag,
                    ethr=current_iter_ethr,
                    avg_diag_iterations=current_iter_avg_diag,
                ))
                current_iter_num = None
                current_iter_energy = None
                current_iter_acc = None

        # 6. Convergence
        elif m := _RE_CONVERGED.search(line):
            output.scf_converged = True
            current_step_converged = True
            output.n_scf_iterations = int(m.group(1))
        elif _RE_NOT_CONVERGED.search(line):
            output.scf_converged = False
            current_step_converged = False

        # 7. Final energy & contributions
        elif m := _RE_FINAL_ENERGY.search(line):
            e = float(m.group(1))
            output.total_energy_ry = e
            current_final_energy = e
        elif m := _RE_ONE_ELECTRON.search(line):
            output.one_electron_ry = float(m.group(1))
        elif m := _RE_HARTREE.search(line):
            output.hartree_ry = float(m.group(1))
        elif m := _RE_XC.search(line):
            output.xc_ry = float(m.group(1))
        elif m := _RE_EWALD.search(line):
            output.ewald_ry = float(m.group(1))
        elif m := _RE_SMEARING.search(line):
            output.smearing_ry = float(m.group(1))

        # 8. Electronic levels
        elif m := _RE_FERMI.search(line):
            output.fermi_energy_ev = float(m.group(1))
        elif m := _RE_BAND_EDGES.search(line):
            output.highest_occupied_ev = float(m.group(1))
            output.lowest_unoccupied_ev = float(m.group(2))
        elif m := _RE_TOTAL_MAG.search(line):
            output.total_magnetization = float(m.group(1))
        elif m := _RE_ABS_MAG.search(line):
            output.absolute_magnetization = float(m.group(1))

        # 9. Forces
        elif "Forces acting on atoms" in line:
            reading_forces = True
            current_forces = []
        elif reading_forces:
            if m := _RE_FORCE_ATOM.search(line):
                atom_idx = int(m.group(1))
                type_idx = int(m.group(2))
                symbol = species_map.get(type_idx, f"Type{type_idx}")
                fx = float(m.group(3))
                fy = float(m.group(4))
                fz = float(m.group(5))
                current_forces.append(ForceAtom(
                    atom_index=atom_idx,
                    symbol=symbol,
                    force=(fx, fy, fz),
                ))
            elif m := _RE_TOTAL_FORCE.search(line):
                current_total_force = float(m.group(1))
                current_scf_corr = float(m.group(2))
                output.total_force = current_total_force
                output.scf_correction_force = current_scf_corr
                reading_forces = False

        # 10. Stress
        elif m := _RE_STRESS_HEADER.search(line):
            reading_stress = True
            stress_lines_remaining = 3
            stress_rows_kbar = []
            stress_pressure = float(m.group(1))
        elif reading_stress and stress_lines_remaining > 0:
            parts = line.split()
            if len(parts) >= 6:
                # 3 components in Ry/bohr^3, 3 in kbar
                s_xx = float(parts[3])
                s_xy = float(parts[4])
                s_xz = float(parts[5])
                stress_rows_kbar.append((s_xx, s_xy, s_xz))
                stress_lines_remaining -= 1
                if stress_lines_remaining == 0:
                    current_stress = StressTensor(
                        tensor_kbar=(
                            stress_rows_kbar[0],
                            stress_rows_kbar[1],
                            stress_rows_kbar[2],
                        ),
                        pressure_kbar=stress_pressure,
                    )
                    output.stress = current_stress
                    reading_stress = False

        # 11. BFGS convergence & step counts
        elif m := _RE_BFGS_CONVERGED.search(line):
            output.bfgs_converged = True
            output.n_bfgs_steps = int(m.group(2))
        elif m := _RE_N_BFGS_STEPS.search(line):
            output.n_bfgs_steps = int(m.group(1))
        elif m := _RE_N_SCF_CYCLES.search(line):
            pass

        # 12. Timing
        elif m := _RE_TIMING.search(line):
            output.cpu_time_seconds = _parse_time(m.group(1))
            output.wall_time_seconds = _parse_time(m.group(2))

        # 13. Warnings & Errors
        elif m := _RE_WARNING.search(line):
            output.warnings.append(m.group(1).strip())
        elif m := _RE_ERROR.search(line):
            output.errors.append(m.group(1).strip())

        # 14. Job done
        elif _RE_JOB_DONE.search(line):
            output.job_done = True

        i += 1

    # End of line loop: commit last pending SCF iteration if any
    if current_iter_num is not None and current_iter_energy is not None and current_iter_acc is not None:
        current_step_scf.append(SCFIteration(
            iteration=current_iter_num,
            total_energy_ry=current_iter_energy,
            accuracy_ry=current_iter_acc,
            diag_method=current_iter_diag,
            ethr=current_iter_ethr,
            avg_diag_iterations=current_iter_avg_diag,
        ))

    # Commit forces to output if present
    if current_forces is not None:
        output.forces = current_forces

    # If we have ionic steps from prior steps plus the final step
    if all_ionic_steps or (output.bfgs_converged or output.n_bfgs_steps):
        if current_step_scf:
            all_ionic_steps.append(IonicStep(
                step=len(all_ionic_steps) + 1,
                scf_iterations=list(current_step_scf),
                converged=output.scf_converged,
                total_energy_ry=output.total_energy_ry,
                forces=output.forces,
                total_force=output.total_force,
                stress=output.stress,
            ))
        output.ionic_steps = all_ionic_steps

    # The top-level scf_iterations is the last (or only) SCF cycle
    output.scf_iterations = current_step_scf

    # Determine exit status
    if output.errors:
        output.exit_status = "error"
    elif output.scf_converged and output.job_done:
        output.exit_status = "converged"
    elif not output.scf_converged and output.job_done:
        output.exit_status = "not_converged"
    elif not output.job_done:
        output.exit_status = "interrupted"
    else:
        output.exit_status = "unknown"

    return output


def read_pw_output(path: str | Path) -> PWOutput:
    """Read and parse a pw.x output file from disk.

    Parameters
    ----------
    path : str or Path
        Path to the output file.

    Returns
    -------
    PWOutput
        Structured representation of the output.
    """
    return parse_pw_output(Path(path).read_text())
