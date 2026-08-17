"""Structural and geometric canonical data models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from qeanalyzer.io.qe_xml import BOHR_TO_ANGSTROM


@dataclass
class QEAtom:
    """A single atom with coordinates in Bohr and Angstrom, plus optional force."""

    symbol: str
    position_bohr: tuple[float, float, float]
    position_angstrom: tuple[float, float, float]
    force_ry_bohr: tuple[float, float, float] | None = None

    @classmethod
    def from_bohr(
        cls,
        symbol: str,
        pos_bohr: tuple[float, float, float],
        force_ry_bohr: tuple[float, float, float] | None = None,
    ) -> QEAtom:
        """Create a QEAtom from coordinates in Bohr."""
        pos_ang = (
            pos_bohr[0] * BOHR_TO_ANGSTROM,
            pos_bohr[1] * BOHR_TO_ANGSTROM,
            pos_bohr[2] * BOHR_TO_ANGSTROM,
        )
        return cls(
            symbol=symbol,
            position_bohr=pos_bohr,
            position_angstrom=pos_ang,
            force_ry_bohr=force_ry_bohr,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert atom to dictionary."""
        return {
            "symbol": self.symbol,
            "position_bohr": list(self.position_bohr),
            "position_angstrom": list(self.position_angstrom),
            "force_ry_bohr": list(self.force_ry_bohr) if self.force_ry_bohr else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QEAtom:
        """Create atom from dictionary."""
        force = tuple(data["force_ry_bohr"]) if data.get("force_ry_bohr") else None
        return cls(
            symbol=data["symbol"],
            position_bohr=tuple(data["position_bohr"]),
            position_angstrom=tuple(data["position_angstrom"]),
            force_ry_bohr=force,
        )


@dataclass
class QEStructure:
    """Atomic structure, cell geometry, volume, forces, and stress."""

    alat_bohr: float | None = None
    alat_angstrom: float | None = None
    cell_vectors_bohr: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    cell_vectors_angstrom: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    volume_bohr3: float | None = None
    volume_angstrom3: float | None = None
    atoms: list[QEAtom] = field(default_factory=list)

    total_force_ry_bohr: float | None = None
    max_force_ry_bohr: float | None = None
    stress_kbar: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    pressure_kbar: float | None = None

    @classmethod
    def from_cell_and_atoms_bohr(
        cls,
        cell_bohr: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] | None,
        atoms_bohr: list[tuple[str, tuple[float, float, float]]],
        alat_bohr: float | None = None,
        forces_ry_bohr: list[tuple[float, float, float]] | None = None,
        stress_kbar: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] | None = None,
        pressure_kbar: float | None = None,
        total_force: float | None = None,
    ) -> QEStructure:
        """Construct a QEStructure from coordinates in atomic units (Bohr)."""
        alat_ang = alat_bohr * BOHR_TO_ANGSTROM if alat_bohr is not None else None

        cell_ang = None
        vol_bohr3 = None
        vol_ang3 = None
        if cell_bohr is not None:
            cell_ang = (
                (
                    cell_bohr[0][0] * BOHR_TO_ANGSTROM,
                    cell_bohr[0][1] * BOHR_TO_ANGSTROM,
                    cell_bohr[0][2] * BOHR_TO_ANGSTROM,
                ),
                (
                    cell_bohr[1][0] * BOHR_TO_ANGSTROM,
                    cell_bohr[1][1] * BOHR_TO_ANGSTROM,
                    cell_bohr[1][2] * BOHR_TO_ANGSTROM,
                ),
                (
                    cell_bohr[2][0] * BOHR_TO_ANGSTROM,
                    cell_bohr[2][1] * BOHR_TO_ANGSTROM,
                    cell_bohr[2][2] * BOHR_TO_ANGSTROM,
                ),
            )
            # Determinant of 3x3 matrix for cell volume
            a, b, c = cell_bohr
            vol_bohr3 = abs(
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
            vol_ang3 = vol_bohr3 * (BOHR_TO_ANGSTROM**3)

        atoms: list[QEAtom] = []
        max_force = 0.0
        has_forces = False
        for i, (sym, pos) in enumerate(atoms_bohr):
            f_vec = forces_ry_bohr[i] if forces_ry_bohr and i < len(forces_ry_bohr) else None
            if f_vec is not None:
                has_forces = True
                norm_f = math.sqrt(f_vec[0] ** 2 + f_vec[1] ** 2 + f_vec[2] ** 2)
                if norm_f > max_force:
                    max_force = norm_f
            atoms.append(QEAtom.from_bohr(sym, pos, f_vec))

        return cls(
            alat_bohr=alat_bohr,
            alat_angstrom=alat_ang,
            cell_vectors_bohr=cell_bohr,
            cell_vectors_angstrom=cell_ang,
            volume_bohr3=vol_bohr3,
            volume_angstrom3=vol_ang3,
            atoms=atoms,
            total_force_ry_bohr=total_force,
            max_force_ry_bohr=max_force if has_forces else None,
            stress_kbar=stress_kbar,
            pressure_kbar=pressure_kbar,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert structure to dictionary."""
        return {
            "alat_bohr": self.alat_bohr,
            "alat_angstrom": self.alat_angstrom,
            "cell_vectors_bohr": (
                [list(row) for row in self.cell_vectors_bohr]
                if self.cell_vectors_bohr
                else None
            ),
            "cell_vectors_angstrom": (
                [list(row) for row in self.cell_vectors_angstrom]
                if self.cell_vectors_angstrom
                else None
            ),
            "volume_bohr3": self.volume_bohr3,
            "volume_angstrom3": self.volume_angstrom3,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "total_force_ry_bohr": self.total_force_ry_bohr,
            "max_force_ry_bohr": self.max_force_ry_bohr,
            "stress_kbar": (
                [list(row) for row in self.stress_kbar] if self.stress_kbar else None
            ),
            "pressure_kbar": self.pressure_kbar,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QEStructure:
        """Construct QEStructure from dictionary."""
        cell_b = (
            tuple(tuple(r) for r in data["cell_vectors_bohr"])
            if data.get("cell_vectors_bohr")
            else None
        )
        cell_a = (
            tuple(tuple(r) for r in data["cell_vectors_angstrom"])
            if data.get("cell_vectors_angstrom")
            else None
        )
        stress = (
            tuple(tuple(r) for r in data["stress_kbar"])
            if data.get("stress_kbar")
            else None
        )
        atoms = [QEAtom.from_dict(d) for d in data.get("atoms", [])]
        return cls(
            alat_bohr=data.get("alat_bohr"),
            alat_angstrom=data.get("alat_angstrom"),
            cell_vectors_bohr=cell_b,
            cell_vectors_angstrom=cell_a,
            volume_bohr3=data.get("volume_bohr3"),
            volume_angstrom3=data.get("volume_angstrom3"),
            atoms=atoms,
            total_force_ry_bohr=data.get("total_force_ry_bohr"),
            max_force_ry_bohr=data.get("max_force_ry_bohr"),
            stress_kbar=stress,
            pressure_kbar=data.get("pressure_kbar"),
        )
