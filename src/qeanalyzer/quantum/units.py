"""Energy-unit helpers for quantum-model interchange.

FCIDUMP does not carry a unit tag.  By convention its one- and two-electron
integrals are Hartree, so all FCIDUMP boundaries in :mod:`qeanalyzer` convert
explicitly to/from Hartree.
"""

from __future__ import annotations

import math

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_RY = 2.0


def normalize_energy_unit(unit: str) -> str:
    key = unit.strip().lower()
    aliases = {
        "ha": "Hartree",
        "hartree": "Hartree",
        "eh": "Hartree",
        "ev": "eV",
        "electronvolt": "eV",
        "electronvolts": "eV",
        "ry": "Ry",
        "rydberg": "Ry",
        "rydbergs": "Ry",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported energy unit {unit!r}; expected Hartree, eV, or Ry") from exc


def energy_to_hartree(value: float, unit: str) -> float:
    unit = normalize_energy_unit(unit)
    if unit == "Hartree":
        return float(value)
    if unit == "eV":
        return float(value) / HARTREE_TO_EV
    return float(value) / HARTREE_TO_RY


def energy_from_hartree(value: float, unit: str) -> float:
    unit = normalize_energy_unit(unit)
    if unit == "Hartree":
        return float(value)
    if unit == "eV":
        return float(value) * HARTREE_TO_EV
    return float(value) * HARTREE_TO_RY


def require_integer_electron_sector(value: float, *, tolerance: float = 1e-8) -> int:
    """Return an integer particle number or reject a genuinely fractional value.

    A smeared periodic DFT occupation is not by itself a choice of many-body
    particle-number sector.  Rounding e.g. 3.63 -> 4 would silently make a
    physical modelling decision, so only values within ``tolerance`` of an
    integer are accepted.
    """
    if not math.isfinite(float(value)):
        raise ValueError("electron count must be finite")
    nearest = int(round(float(value)))
    if abs(float(value) - nearest) > tolerance:
        raise ValueError(
            f"Active electron count {value:.12g} is fractional. "
            "Choose an explicit integer particle-number sector before FCI/ADAPT/FCIDUMP export."
        )
    if nearest < 0:
        raise ValueError("electron count must be non-negative")
    return nearest
