"""Parse Quantum ESPRESSO structured XML data files.

Parses ``data-file-schema.xml`` (produced in ``prefix.save/``), extracting
structured electronic, structural, convergence, and energetic ground-truth data.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# -- Physical constants ------------------------------------------------------

HARTREE_TO_RY: float = 2.0
HARTREE_TO_EV: float = 27.211386245988
BOHR_TO_ANGSTROM: float = 0.529177210903
# 1 Hartree / Bohr^3 to kbar: 1 Ha = 4.3597447222071e-18 J, 1 Bohr = 5.29177210903e-11 m
# 1 Ha/Bohr^3 = 2.942102648438959e13 Pa = 294210.2648438959 bar = 29421.02648438959 kbar
HARTREE_BOHR3_TO_KBAR: float = 29421.02648438959


# -- Data classes ------------------------------------------------------------

@dataclass
class QEXMLKPoint:
    """A single k-point with eigenvalues and occupations from XML."""

    point: tuple[float, float, float]
    weight: float
    eigenvalues_hartree: list[float]
    eigenvalues_ev: list[float]
    occupations: list[float]


@dataclass
class QEXMLStructure:
    """Atomic structure and lattice vectors extracted from XML."""

    alat_bohr: float | None
    cell_vectors_bohr: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    positions_bohr: list[tuple[str, tuple[float, float, float]]]  # (symbol, (x, y, z))


@dataclass
class QEXMLOutput:
    """Complete parsed representation of a QE data-file-schema.xml file."""

    # Schema and provenance
    schema_version: str | None = None
    creator_name: str | None = None
    creator_version: str | None = None
    job_title: str | None = None
    nprocs: int | None = None
    nthreads: int | None = None

    # Input parameters echo
    calculation: str | None = None
    prefix: str | None = None
    ecutwfc_hartree: float | None = None
    ecutrho_hartree: float | None = None
    conv_thr: float | None = None
    mixing_beta: float | None = None

    # Convergence
    scf_converged: bool = False
    n_scf_steps: int | None = None
    scf_error: float | None = None
    opt_converged: bool | None = None
    n_opt_steps: int | None = None

    # Total energies (Hartree, Ry, eV)
    total_energy_hartree: float | None = None
    one_electron_hartree: float | None = None
    hartree_energy_hartree: float | None = None
    xc_energy_hartree: float | None = None
    ewald_energy_hartree: float | None = None
    demet_energy_hartree: float | None = None

    # Electronic structure
    n_electrons: float | None = None
    n_bands: int | None = None
    n_kpoints: int | None = None
    lsda: bool = False
    noncolin: bool = False
    spinorbit: bool = False

    fermi_energy_hartree: float | None = None
    highest_occupied_hartree: float | None = None
    lowest_unoccupied_hartree: float | None = None

    kpoints: list[QEXMLKPoint] = field(default_factory=list)

    # Geometry & properties
    initial_structure: QEXMLStructure | None = None
    final_structure: QEXMLStructure | None = None
    forces_hartree_bohr: list[tuple[float, float, float]] | None = None
    stress_kbar: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    pressure_kbar: float | None = None

    # -- Convenience properties ----------------------------------------------

    @property
    def total_energy_ry(self) -> float | None:
        """Total energy in Rydberg."""
        if self.total_energy_hartree is None:
            return None
        return self.total_energy_hartree * HARTREE_TO_RY

    @property
    def total_energy_ev(self) -> float | None:
        """Total energy in electron-volts."""
        if self.total_energy_hartree is None:
            return None
        return self.total_energy_hartree * HARTREE_TO_EV

    @property
    def fermi_energy_ev(self) -> float | None:
        """Fermi energy in electron-volts."""
        if self.fermi_energy_hartree is None:
            return None
        return self.fermi_energy_hartree * HARTREE_TO_EV

    @property
    def highest_occupied_ev(self) -> float | None:
        """Highest occupied level in electron-volts."""
        if self.highest_occupied_hartree is None:
            return None
        return self.highest_occupied_hartree * HARTREE_TO_EV

    @property
    def lowest_unoccupied_ev(self) -> float | None:
        """Lowest unoccupied level in electron-volts."""
        if self.lowest_unoccupied_hartree is None:
            return None
        return self.lowest_unoccupied_hartree * HARTREE_TO_EV

    @property
    def band_gap_ev(self) -> float | None:
        """Fundamental band gap in eV, or None if metallic or not available."""
        if self.highest_occupied_ev is not None and self.lowest_unoccupied_ev is not None:
            gap = self.lowest_unoccupied_ev - self.highest_occupied_ev
            return max(0.0, gap)
        return None

    @property
    def forces_ry_bohr(self) -> list[tuple[float, float, float]] | None:
        """Forces on atoms in Ry/Bohr (Ry/au)."""
        if self.forces_hartree_bohr is None:
            return None
        return [
            (fx * HARTREE_TO_RY, fy * HARTREE_TO_RY, fz * HARTREE_TO_RY)
            for fx, fy, fz in self.forces_hartree_bohr
        ]


# -- Helper functions --------------------------------------------------------

def _clean_tag(elem: ET.Element) -> str:
    """Return local tag name without XML namespace."""
    tag = elem.tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find(elem: ET.Element | None, tag: str) -> ET.Element | None:
    """Find a child element ignoring namespaces."""
    if elem is None:
        return None
    for child in elem:
        if _clean_tag(child) == tag:
            return child
    return None


def _findall(elem: ET.Element | None, tag: str) -> list[ET.Element]:
    """Find all matching children ignoring namespaces."""
    if elem is None:
        return []
    return [child for child in elem if _clean_tag(child) == tag]


def _text(elem: ET.Element | None) -> str | None:
    """Get stripped text of an element or None."""
    if elem is None or elem.text is None:
        return None
    s = elem.text.strip()
    return s if s else None


def _float(elem: ET.Element | None) -> float | None:
    """Parse float from element text."""
    s = _text(elem)
    if s is None:
        return None
    try:
        return float(s.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _int(elem: ET.Element | None) -> int | None:
    """Parse int from element text."""
    s = _text(elem)
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _bool(elem: ET.Element | None) -> bool:
    """Parse boolean from element text."""
    s = _text(elem)
    if s is None:
        return False
    return s.lower() in ("true", "1", "t")


def _parse_structure(elem: ET.Element | None) -> QEXMLStructure | None:
    """Parse an atomic_structure XML element."""
    if elem is None:
        return None

    alat = None
    if "alat" in elem.attrib:
        try:
            alat = float(elem.attrib["alat"])
        except ValueError:
            pass

    cell_elem = _find(elem, "cell")
    if cell_elem is None:
        return None

    a1_elem = _find(cell_elem, "a1")
    a2_elem = _find(cell_elem, "a2")
    a3_elem = _find(cell_elem, "a3")

    if a1_elem is None or a2_elem is None or a3_elem is None:
        return None

    try:
        a1_parts = [float(x) for x in (_text(a1_elem) or "").split()]
        a2_parts = [float(x) for x in (_text(a2_elem) or "").split()]
        a3_parts = [float(x) for x in (_text(a3_elem) or "").split()]
        vectors = (
            (a1_parts[0], a1_parts[1], a1_parts[2]),
            (a2_parts[0], a2_parts[1], a2_parts[2]),
            (a3_parts[0], a3_parts[1], a3_parts[2]),
        )
    except (IndexError, ValueError):
        return None

    positions_elem = _find(elem, "atomic_positions")
    positions: list[tuple[str, tuple[float, float, float]]] = []
    if positions_elem is not None:
        for atom in _findall(positions_elem, "atom"):
            name = atom.attrib.get("name", "X")
            coords_str = _text(atom) or ""
            try:
                coords = [float(x) for x in coords_str.split()]
                if len(coords) >= 3:
                    positions.append((name, (coords[0], coords[1], coords[2])))
            except ValueError:
                continue

    return QEXMLStructure(
        alat_bohr=alat,
        cell_vectors_bohr=vectors,
        positions_bohr=positions,
    )


# -- Main Parser -------------------------------------------------------------

def parse_qe_xml(xml_content: str) -> QEXMLOutput:
    """Parse a Quantum ESPRESSO data-file-schema.xml string.

    Parameters
    ----------
    xml_content : str
        XML text content.

    Returns
    -------
    QEXMLOutput
        Parsed structured record.
    """
    root = ET.fromstring(xml_content)
    result = QEXMLOutput()

    # Schema version
    result.schema_version = root.attrib.get("SCHEMA_VERSION") or root.attrib.get("qes_version")
    if result.schema_version and not result.schema_version.startswith("1."):
        warnings.warn(
            f"Unrecognised or unverified QE XML schema version: {result.schema_version}",
            UserWarning,
            stacklevel=2,
        )

    # 1. General Info
    gen_elem = _find(root, "general_info")
    if gen_elem is not None:
        creator_elem = _find(gen_elem, "creator")
        if creator_elem is not None:
            result.creator_name = creator_elem.attrib.get("NAME")
            result.creator_version = creator_elem.attrib.get("VERSION")
        result.job_title = _text(_find(gen_elem, "job_title"))

    # 2. Parallel Info
    par_elem = _find(root, "parallel_info")
    if par_elem is not None:
        result.nprocs = _int(_find(par_elem, "nprocs"))
        result.nthreads = _int(_find(par_elem, "nthreads"))

    # 3. Input
    input_elem = _find(root, "input")
    if input_elem is not None:
        ctrl_elem = _find(input_elem, "control_variables")
        if ctrl_elem is not None:
            result.calculation = _text(_find(ctrl_elem, "calculation"))
            result.prefix = _text(_find(ctrl_elem, "prefix"))

        basis_elem = _find(input_elem, "basis")
        if basis_elem is not None:
            result.ecutwfc_hartree = _float(_find(basis_elem, "ecutwfc"))
            result.ecutrho_hartree = _float(_find(basis_elem, "ecutrho"))

        elec_ctrl = _find(input_elem, "electron_control")
        if elec_ctrl is not None:
            result.conv_thr = _float(_find(elec_ctrl, "conv_thr"))
            result.mixing_beta = _float(_find(elec_ctrl, "mixing_beta"))

        result.initial_structure = _parse_structure(_find(input_elem, "atomic_structure"))

    # 4. Output
    output_elem = _find(root, "output")
    if output_elem is not None:
        # Convergence
        conv_elem = _find(output_elem, "convergence_info")
        if conv_elem is not None:
            scf_conv = _find(conv_elem, "scf_conv")
            if scf_conv is not None:
                result.scf_converged = True
                result.n_scf_steps = _int(_find(scf_conv, "n_scf_steps"))
                result.scf_error = _float(_find(scf_conv, "scf_error"))

            opt_conv = _find(conv_elem, "opt_conv")
            if opt_conv is not None:
                result.opt_converged = True
                result.n_opt_steps = _int(_find(opt_conv, "n_opt_steps"))

        # Total energy
        tot_elem = _find(output_elem, "total_energy")
        if tot_elem is not None:
            result.total_energy_hartree = _float(_find(tot_elem, "etot"))
            result.one_electron_hartree = _float(_find(tot_elem, "eband"))
            result.hartree_energy_hartree = _float(_find(tot_elem, "ehart"))
            result.xc_energy_hartree = _float(_find(tot_elem, "etxc"))
            result.ewald_energy_hartree = _float(_find(tot_elem, "ewald"))
            result.demet_energy_hartree = _float(_find(tot_elem, "demet"))

        # Band structure
        band_elem = _find(output_elem, "band_structure")
        if band_elem is not None:
            result.lsda = _bool(_find(band_elem, "lsda"))
            result.noncolin = _bool(_find(band_elem, "noncolin"))
            result.spinorbit = _bool(_find(band_elem, "spinorbit"))
            result.n_electrons = _float(_find(band_elem, "num_of_electrons"))
            result.n_bands = _int(_find(band_elem, "num_of_bands"))
            result.n_kpoints = _int(_find(band_elem, "num_of_k_points"))

            result.fermi_energy_hartree = _float(_find(band_elem, "fermi_energy"))
            result.highest_occupied_hartree = _float(_find(band_elem, "highestOccupiedLevel"))
            result.lowest_unoccupied_hartree = _float(_find(band_elem, "lowestUnoccupiedLevel"))

            for ks_elem in _findall(band_elem, "ks_energies"):
                kp_elem = _find(ks_elem, "k_point")
                eig_elem = _find(ks_elem, "eigenvalues")
                occ_elem = _find(ks_elem, "occupations")

                kpt = (0.0, 0.0, 0.0)
                weight = 1.0
                if kp_elem is not None:
                    weight = float(kp_elem.attrib.get("weight", "1.0"))
                    try:
                        k_coords = [float(x) for x in (_text(kp_elem) or "").split()]
                        if len(k_coords) >= 3:
                            kpt = (k_coords[0], k_coords[1], k_coords[2])
                    except ValueError:
                        pass

                eigs_ha: list[float] = []
                eigs_ev: list[float] = []
                if eig_elem is not None:
                    try:
                        eigs_ha = [float(x) for x in (_text(eig_elem) or "").split()]
                        eigs_ev = [e * HARTREE_TO_EV for e in eigs_ha]
                    except ValueError:
                        pass

                occs: list[float] = []
                if occ_elem is not None:
                    try:
                        occs = [float(x) for x in (_text(occ_elem) or "").split()]
                    except ValueError:
                        pass

                result.kpoints.append(QEXMLKPoint(
                    point=kpt,
                    weight=weight,
                    eigenvalues_hartree=eigs_ha,
                    eigenvalues_ev=eigs_ev,
                    occupations=occs,
                ))

        # Forces
        forces_elem = _find(output_elem, "forces")
        if forces_elem is not None:
            f_text = _text(forces_elem) or ""
            forces_list: list[tuple[float, float, float]] = []
            for line in f_text.split("\n"):
                parts = [float(x) for x in line.split()]
                if len(parts) >= 3:
                    forces_list.append((parts[0], parts[1], parts[2]))
            if forces_list:
                result.forces_hartree_bohr = forces_list

        # Stress
        stress_elem = _find(output_elem, "stress")
        if stress_elem is not None:
            s_text = _text(stress_elem) or ""
            stress_rows_ha: list[list[float]] = []
            for line in s_text.split("\n"):
                parts = [float(x) for x in line.split()]
                if len(parts) >= 3:
                    stress_rows_ha.append([parts[0], parts[1], parts[2]])
            if len(stress_rows_ha) >= 3:
                # Convert from Hartree/Bohr^3 to kbar
                s_kbar = (
                    (
                        stress_rows_ha[0][0] * HARTREE_BOHR3_TO_KBAR,
                        stress_rows_ha[0][1] * HARTREE_BOHR3_TO_KBAR,
                        stress_rows_ha[0][2] * HARTREE_BOHR3_TO_KBAR,
                    ),
                    (
                        stress_rows_ha[1][0] * HARTREE_BOHR3_TO_KBAR,
                        stress_rows_ha[1][1] * HARTREE_BOHR3_TO_KBAR,
                        stress_rows_ha[1][2] * HARTREE_BOHR3_TO_KBAR,
                    ),
                    (
                        stress_rows_ha[2][0] * HARTREE_BOHR3_TO_KBAR,
                        stress_rows_ha[2][1] * HARTREE_BOHR3_TO_KBAR,
                        stress_rows_ha[2][2] * HARTREE_BOHR3_TO_KBAR,
                    ),
                )
                result.stress_kbar = s_kbar
                # Pressure P = -1/3 Tr(sigma)
                result.pressure_kbar = -(s_kbar[0][0] + s_kbar[1][1] + s_kbar[2][2]) / 3.0

        # Final structure (for relax/vc-relax)
        final_struct_elem = _find(output_elem, "atomic_structure")
        if final_struct_elem is not None:
            result.final_structure = _parse_structure(final_struct_elem)

    return result


def read_qe_xml(path: str | Path) -> QEXMLOutput:
    """Read and parse a Quantum ESPRESSO data-file-schema.xml file from disk.

    Parameters
    ----------
    path : str or Path
        Path to the XML file.

    Returns
    -------
    QEXMLOutput
        Parsed structured record.
    """
    return parse_qe_xml(Path(path).read_text())
