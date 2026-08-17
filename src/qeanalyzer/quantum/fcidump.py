"""Restricted real FCIDUMP reader/writer with an explicit Hartree boundary."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from qeanalyzer.quantum.hamiltonian import MaterialHamiltonian
from qeanalyzer.quantum.units import energy_to_hartree, require_integer_electron_sector

_HEADER_START = re.compile(r"^\s*&(FCI|FCIDUMP)\b", re.IGNORECASE)
_HEADER_END = re.compile(r"&END\b|/", re.IGNORECASE)


def _fortran_float(token: str) -> float:
    value = float(token.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError("FCIDUMP coefficients must be finite")
    return value


def _pair_index(i: int, j: int) -> int:
    """Packed lower-triangle index for zero-based i>=j."""
    return i * (i + 1) // 2 + j


def write_fcidump(
    hamiltonian: MaterialHamiltonian,
    path: str | Path | None = None,
    orbsym: list[int] | None = None,
    isym: int = 1,
    tolerance: float = 1e-12,
) -> str:
    """Export a restricted real Hamiltonian to FCIDUMP.

    FCIDUMP has no unit field.  This writer therefore converts every energy
    quantity to **Hartree**, regardless of ``hamiltonian.energy_unit``.  This
    makes files interoperable with PySCF, NECI, Dice, ``clifford_qc`` and other
    conventional consumers.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if not hamiltonian.is_hermitian():
        raise ValueError("Hamiltonian must satisfy restricted real integral symmetries")
    norb = hamiltonian.n_orbitals
    nelec = require_integer_electron_sector(hamiltonian.n_electrons)
    ms2 = int(hamiltonian.spin)
    if abs(ms2) > nelec or (nelec + ms2) % 2:
        raise ValueError("NELEC and MS2 do not define an integer alpha/beta sector")
    sym = list(orbsym) if orbsym is not None else [1] * norb
    if len(sym) != norb:
        raise ValueError("ORBSYM must contain exactly NORB entries")

    lines = [
        "&FCI",
        f" NORB={norb},",
        f" NELEC={nelec},",
        f" MS2={ms2},",
        f" ORBSYM={','.join(str(int(x)) for x in sym)},",
        f" ISYM={int(isym)},",
        " IUHF=0,",
        "/",
    ]

    # h2[p][q][r][s] is exactly chemist (pq|rs).  Write one representative
    # from each eightfold symmetry class.
    for i in range(norb):
        for j in range(i + 1):
            ij = _pair_index(i, j)
            for k in range(norb):
                for l in range(k + 1):
                    if ij < _pair_index(k, l):
                        continue
                    value = energy_to_hartree(hamiltonian.h2[i][j][k][l], hamiltonian.energy_unit)
                    if abs(value) > tolerance:
                        lines.append(f"{value:23.16E} {i+1:4d} {j+1:4d} {k+1:4d} {l+1:4d}")

    for i in range(norb):
        for j in range(i + 1):
            value = energy_to_hartree(hamiltonian.h1[i][j], hamiltonian.energy_unit)
            if abs(value) > tolerance:
                lines.append(f"{value:23.16E} {i+1:4d} {j+1:4d} {0:4d} {0:4d}")

    core = energy_to_hartree(hamiltonian.constant, hamiltonian.energy_unit)
    lines.append(f"{core:23.16E} {0:4d} {0:4d} {0:4d} {0:4d}")
    text = "\n".join(lines) + "\n"
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    return text


def _parse_header(lines: list[str]) -> tuple[dict[str, str], int]:
    header_parts: list[str] = []
    start = None
    for idx, original in enumerate(lines):
        line = original.split("!", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        if start is None and _HEADER_START.match(line) is None:
            raise ValueError("FCIDUMP must start with &FCI or &FCIDUMP")
        start = 0
        header_parts.append(line)
        if _HEADER_END.search(line):
            header = " ".join(header_parts)
            data: dict[str, str] = {}
            for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,&/]+(?:,[^A-Za-z&/][^&/]*)?)", header):
                data[match.group(1).upper()] = match.group(2).strip().rstrip(",")
            # Simpler scalar extraction overrides the permissive regex above.
            for key in ("NORB", "NELEC", "MS2", "ISYM", "IUHF"):
                m = re.search(rf"\b{key}\s*=\s*([^,\s/&]+)", header, re.IGNORECASE)
                if m:
                    data[key] = m.group(1)
            m = re.search(r"\bORBSYM\s*=\s*(.*?)(?=\bISYM\s*=|\bIUHF\s*=|&END\b|/|$)", header, re.IGNORECASE)
            if m:
                data["ORBSYM"] = m.group(1).strip(" ,")
            return data, idx + 1
    raise ValueError("FCIDUMP header is missing a terminator")


def parse_fcidump(text: str) -> MaterialHamiltonian:
    """Parse restricted real FCIDUMP values as Hartree.

    The returned ``MaterialHamiltonian.energy_unit`` is always ``"Hartree"``;
    FCIDUMP itself cannot communicate any alternative unit.
    """
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        raise ValueError("Empty FCIDUMP content")
    header, body_start = _parse_header(lines)
    if "NORB" not in header:
        raise ValueError("NORB parameter missing in FCIDUMP header")
    norb = int(header["NORB"])
    nelec = int(header.get("NELEC", "0"))
    ms2 = int(header.get("MS2", "0"))
    iuhf = int(header.get("IUHF", "0"))
    if iuhf != 0:
        raise NotImplementedError("Unrestricted FCIDUMP (IUHF=1) is not supported")
    if norb <= 0 or not 0 <= nelec <= 2 * norb:
        raise ValueError("Invalid NORB/NELEC sector")
    if abs(ms2) > nelec or (nelec + ms2) % 2:
        raise ValueError("NELEC and MS2 are incompatible")

    h1 = [[0.0] * norb for _ in range(norb)]
    h2 = [[[[0.0 for _ in range(norb)] for _ in range(norb)] for _ in range(norb)] for _ in range(norb)]
    constant = 0.0

    def two_perms(i: int, j: int, k: int, l: int) -> set[tuple[int, int, int, int]]:
        return {
            (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
            (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i),
        }

    for line_number, original in enumerate(lines[body_start:], start=body_start + 1):
        line = original.split("!", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"FCIDUMP line {line_number} must contain coefficient and four indices")
        value = _fortran_float(fields[0])
        try:
            i, j, k, l = (int(x) for x in fields[1:])
        except ValueError as exc:
            raise ValueError(f"FCIDUMP line {line_number} has non-integer indices") from exc
        if any(idx < 0 or idx > norb for idx in (i, j, k, l)):
            raise ValueError(f"FCIDUMP line {line_number} index outside [0,NORB]")
        if (i, j, k, l) == (0, 0, 0, 0):
            constant = value
        elif i > 0 and j > 0 and k == 0 and l == 0:
            h1[i - 1][j - 1] = value
            h1[j - 1][i - 1] = value
        elif all(idx > 0 for idx in (i, j, k, l)):
            for p, q, r, s in two_perms(i - 1, j - 1, k - 1, l - 1):
                h2[p][q][r][s] = value
        else:
            raise ValueError(f"FCIDUMP line {line_number} uses an invalid zero sentinel pattern")

    orbsym = [int(x) for x in re.split(r"[\s,]+", header.get("ORBSYM", "").strip()) if x]
    return MaterialHamiltonian(
        n_orbitals=norb,
        n_electrons=float(nelec),
        constant=constant,
        spin=ms2,
        energy_unit="Hartree",
        h1=h1,
        h2=h2,
        metadata={
            "source": "fcidump",
            "model_kind": "explicit_integrals",
            "integral_convention": "chemist_(pq|rs)",
            "orbsym": orbsym or [1] * norb,
            "isym": int(header.get("ISYM", "1")),
            "fcidump_energy_unit": "Hartree",
        },
    )


def read_fcidump(path: str | Path) -> MaterialHamiltonian:
    return parse_fcidump(Path(path).read_text(encoding="utf-8"))
