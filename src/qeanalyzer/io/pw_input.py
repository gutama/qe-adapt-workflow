"""Parse and write Quantum ESPRESSO pw.x input files.

Handles Fortran namelist syntax, atomic species/positions,
k-point specifications, and cell parameters.  Values are converted
to native Python types on read and formatted back to valid Fortran
namelist syntax on write, supporting deterministic round-trips.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# -- Constants ---------------------------------------------------------------

# Namelist names recognised by pw.x, in canonical order.
NAMELIST_ORDER: tuple[str, ...] = (
    "CONTROL", "SYSTEM", "ELECTRONS", "IONS", "CELL",
)

# Card names recognised by pw.x, in canonical order.
CARD_ORDER: tuple[str, ...] = (
    "ATOMIC_SPECIES",
    "ATOMIC_POSITIONS",
    "K_POINTS",
    "CELL_PARAMETERS",
    "CONSTRAINTS",
    "OCCUPATIONS",
    "ATOMIC_FORCES",
)

# Matches Fortran-style floats: 1.0, .5, 1.0d-8, 2.5D+3, 3e2, etc.
_FORTRAN_FLOAT_RE = re.compile(
    r"^[+-]?(\d+\.?\d*|\.\d+)([dDeE][+-]?\d+)?$"
)


# -- Data classes ------------------------------------------------------------

@dataclass
class AtomicSpecies:
    """A single atomic species: symbol, mass (amu), pseudopotential file."""

    symbol: str
    mass: float
    pseudo_file: str


@dataclass
class AtomicPosition:
    """A single atomic position with optional if_pos constraints."""

    symbol: str
    position: tuple[float, float, float]
    if_pos: tuple[int, int, int] | None = None


@dataclass
class KPoints:
    """K-point specification.

    For ``option='automatic'``, *grid* and *offsets* are set.
    For ``option='gamma'``, no extra data is needed.
    For explicit lists (``'crystal'``, ``'tpiba'``, etc.), *nk* and
    *points* are set.
    """

    option: str
    grid: tuple[int, int, int] | None = None
    offsets: tuple[int, int, int] | None = None
    nk: int | None = None
    points: list[tuple[float, float, float, float]] | None = None


@dataclass
class CellParameters:
    """Unit-cell lattice vectors."""

    option: str  # "bohr", "angstrom", or "alat"
    vectors: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


@dataclass
class PWInput:
    """Complete parsed representation of a pw.x input file.

    Namelists are stored as ``{NAME: {key: value, ...}}``.  All namelist
    keys are lower-cased; values are converted to native Python types.
    """

    namelists: dict[str, dict[str, Any]] = field(default_factory=dict)
    atomic_species: list[AtomicSpecies] = field(default_factory=list)
    atomic_positions_option: str = "alat"
    atomic_positions: list[AtomicPosition] = field(default_factory=list)
    kpoints: KPoints | None = None
    cell_parameters: CellParameters | None = None

    # -- Convenience properties ----------------------------------------------

    @property
    def calculation(self) -> str:
        """Calculation type (``'scf'`` by default)."""
        return self.namelists.get("CONTROL", {}).get("calculation", "scf")

    @property
    def nat(self) -> int | None:
        """Number of atoms."""
        return self.namelists.get("SYSTEM", {}).get("nat")

    @property
    def ntyp(self) -> int | None:
        """Number of atomic types."""
        return self.namelists.get("SYSTEM", {}).get("ntyp")

    @property
    def ecutwfc(self) -> float | None:
        """Plane-wave kinetic-energy cutoff (Ry)."""
        return self.namelists.get("SYSTEM", {}).get("ecutwfc")

    @property
    def ecutrho(self) -> float | None:
        """Charge-density cutoff (Ry)."""
        return self.namelists.get("SYSTEM", {}).get("ecutrho")


# -- Value helpers -----------------------------------------------------------

def _strip_comment(line: str) -> str:
    """Remove a trailing ``!`` comment, respecting single-quoted strings."""
    in_quote = False
    for i, ch in enumerate(line):
        if ch == "'":
            in_quote = not in_quote
        elif ch == "!" and not in_quote:
            return line[:i]
    return line


def _find_namelist_terminator(line: str) -> int | None:
    """Return index of the ``/`` that ends a namelist, or *None*."""
    in_quote = False
    for i, ch in enumerate(line):
        if ch == "'":
            in_quote = not in_quote
        elif ch == "/" and not in_quote:
            return i
    return None


def _parse_value(raw: str) -> bool | int | float | str:
    """Convert a raw Fortran namelist value to a Python object.

    Recognises booleans (``.true.``/``.false.``), integers, standard and
    Fortran-style floats (``1.0d-8``), and single- or double-quoted
    strings.
    """
    s = raw.strip()

    # Booleans
    if s.lower() in (".true.", ".t."):
        return True
    if s.lower() in (".false.", ".f."):
        return False

    # Quoted strings
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1]
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]

    # Fortran float with d/D exponent
    if "d" in s.lower() and _FORTRAN_FLOAT_RE.match(s):
        return float(s.lower().replace("d", "e"))

    # Integer
    try:
        return int(s)
    except ValueError:
        pass

    # Float
    try:
        return float(s)
    except ValueError:
        pass

    # Fallback – return the raw token.
    return s


def _format_value(value: Any) -> str:
    """Format a Python value back to Fortran namelist syntax."""
    # bool must be checked before int (bool is a subclass of int).
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


# -- Namelist parsing --------------------------------------------------------

def _parse_namelist_body(body: str) -> dict[str, Any]:
    """Parse the body text of a Fortran namelist into a dict.

    Entries may be separated by commas or newlines.  Commas inside
    single-quoted strings are handled correctly.
    """
    result: dict[str, Any] = {}

    # Split into entries on commas/newlines, respecting quotes.
    entries: list[str] = []
    current: list[str] = []
    in_quote = False
    for ch in body:
        if ch == "'":
            in_quote = not in_quote
            current.append(ch)
        elif ch in (",", "\n") and not in_quote:
            entries.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        entries.append("".join(current))

    for entry in entries:
        entry = entry.strip()
        if not entry or entry == "/":
            continue
        if "=" not in entry:
            continue
        key_part, val_part = entry.split("=", 1)
        key = key_part.strip().lower()
        result[key] = _parse_value(val_part)

    return result


# -- Card parsing ------------------------------------------------------------

def _parse_card_option(line: str, card_name: str) -> str:
    """Extract the option keyword from a card header line.

    Handles ``CARD_NAME option``, ``CARD_NAME {option}``, and
    ``CARD_NAME (option)`` forms.
    """
    rest = line[len(card_name):].strip()
    for ch in "{}()":
        rest = rest.replace(ch, "")
    return rest.strip().lower()


def _parse_atomic_species_lines(lines: list[str]) -> list[AtomicSpecies]:
    species: list[AtomicSpecies] = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            species.append(AtomicSpecies(
                symbol=parts[0],
                mass=float(parts[1]),
                pseudo_file=parts[2],
            ))
    return species


def _parse_atomic_positions_lines(lines: list[str]) -> list[AtomicPosition]:
    positions: list[AtomicPosition] = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            if_pos = None
            if len(parts) >= 7:
                if_pos = (int(parts[4]), int(parts[5]), int(parts[6]))
            positions.append(AtomicPosition(
                symbol=parts[0],
                position=(float(parts[1]), float(parts[2]), float(parts[3])),
                if_pos=if_pos,
            ))
    return positions


def _parse_kpoints_lines(option: str, lines: list[str]) -> KPoints:
    if option == "gamma":
        return KPoints(option="gamma")

    if option == "automatic":
        parts = lines[0].split()
        return KPoints(
            option="automatic",
            grid=(int(parts[0]), int(parts[1]), int(parts[2])),
            offsets=(int(parts[3]), int(parts[4]), int(parts[5])),
        )

    # Explicit list: crystal, crystal_b, tpiba, tpiba_b
    nk = int(lines[0].strip())
    points: list[tuple[float, float, float, float]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            points.append((
                float(parts[0]), float(parts[1]),
                float(parts[2]), float(parts[3]),
            ))
    return KPoints(option=option, nk=nk, points=points)


def _parse_cell_parameters_lines(
    option: str,
    lines: list[str],
) -> CellParameters:
    vectors: list[tuple[float, float, float]] = []
    for line in lines[:3]:
        parts = line.split()
        vectors.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return CellParameters(
        option=option or "bohr",
        vectors=(vectors[0], vectors[1], vectors[2]),
    )


# -- Main parser -------------------------------------------------------------

def parse_pw_input(text: str) -> PWInput:
    """Parse a complete pw.x input file.

    Parameters
    ----------
    text : str
        Full text content of the input file.

    Returns
    -------
    PWInput
        Structured representation of the input.
    """
    result = PWInput()

    raw_lines = text.replace("\r\n", "\n").split("\n")
    lines = [_strip_comment(line) for line in raw_lines]

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            i += 1
            continue

        # ---- Namelist start ------------------------------------------------
        if stripped.startswith("&"):
            namelist_name = stripped[1:].strip().upper()
            body_lines: list[str] = []
            i += 1
            while i < len(lines):
                sl = lines[i].strip()
                term = _find_namelist_terminator(sl)
                if term is not None:
                    before = sl[:term].strip()
                    if before:
                        body_lines.append(before)
                    i += 1
                    break
                body_lines.append(sl)
                i += 1
            result.namelists[namelist_name] = _parse_namelist_body(
                "\n".join(body_lines),
            )
            continue

        # ---- Card start ----------------------------------------------------
        upper = stripped.upper()
        matched_card: str | None = None
        for card_name in CARD_ORDER:
            if upper.startswith(card_name):
                matched_card = card_name
                break

        if matched_card is not None:
            option = _parse_card_option(stripped, matched_card)
            i += 1
            card_lines: list[str] = []
            while i < len(lines):
                sl = lines[i].strip()
                if not sl:
                    i += 1
                    continue
                su = sl.upper()
                if su.startswith("&"):
                    break
                if any(su.startswith(cn) for cn in CARD_ORDER):
                    break
                card_lines.append(sl)
                i += 1

            if matched_card == "ATOMIC_SPECIES":
                result.atomic_species = _parse_atomic_species_lines(
                    card_lines,
                )
            elif matched_card == "ATOMIC_POSITIONS":
                result.atomic_positions_option = option or "alat"
                result.atomic_positions = _parse_atomic_positions_lines(
                    card_lines,
                )
            elif matched_card == "K_POINTS":
                result.kpoints = _parse_kpoints_lines(
                    option or "tpiba", card_lines,
                )
            elif matched_card == "CELL_PARAMETERS":
                result.cell_parameters = _parse_cell_parameters_lines(
                    option, card_lines,
                )
            continue

        i += 1

    return result


def read_pw_input(path: str | Path) -> PWInput:
    """Read and parse a pw.x input file from disk.

    Parameters
    ----------
    path : str or Path
        Path to the input file.

    Returns
    -------
    PWInput
        Structured representation of the input.
    """
    return parse_pw_input(Path(path).read_text())


# -- Writer ------------------------------------------------------------------

def write_pw_input(pw: PWInput) -> str:
    """Serialize a *PWInput* back to pw.x input file format.

    Parameters
    ----------
    pw : PWInput
        The input object to serialize.

    Returns
    -------
    str
        Valid pw.x input file content.
    """
    sections: list[str] = []

    # Namelists in canonical order.
    for name in NAMELIST_ORDER:
        if name not in pw.namelists:
            continue
        nl = pw.namelists[name]
        block = [f"&{name}"]
        for key in sorted(nl):
            block.append(f"  {key} = {_format_value(nl[key])},")
        block.append("/")
        sections.append("\n".join(block))

    # ATOMIC_SPECIES
    if pw.atomic_species:
        block = ["ATOMIC_SPECIES"]
        for sp in pw.atomic_species:
            block.append(f"  {sp.symbol}  {sp.mass}  {sp.pseudo_file}")
        sections.append("\n".join(block))

    # ATOMIC_POSITIONS
    if pw.atomic_positions:
        block = [f"ATOMIC_POSITIONS {{{pw.atomic_positions_option}}}"]
        for ap in pw.atomic_positions:
            x, y, z = ap.position
            line = f"  {ap.symbol}  {x:.10f}  {y:.10f}  {z:.10f}"
            if ap.if_pos is not None:
                line += f"  {ap.if_pos[0]}  {ap.if_pos[1]}  {ap.if_pos[2]}"
            block.append(line)
        sections.append("\n".join(block))

    # K_POINTS
    if pw.kpoints is not None:
        kp = pw.kpoints
        if kp.option == "gamma":
            sections.append("K_POINTS {gamma}")
        elif kp.option == "automatic":
            g = kp.grid or (1, 1, 1)
            o = kp.offsets or (0, 0, 0)
            sections.append("\n".join([
                "K_POINTS {automatic}",
                f"  {g[0]}  {g[1]}  {g[2]}  {o[0]}  {o[1]}  {o[2]}",
            ]))
        else:
            block = [f"K_POINTS {{{kp.option}}}"]
            block.append(f"  {kp.nk}")
            for pt in kp.points or []:
                block.append(
                    f"  {pt[0]:.10f}  {pt[1]:.10f}"
                    f"  {pt[2]:.10f}  {pt[3]:.10f}"
                )
            sections.append("\n".join(block))

    # CELL_PARAMETERS
    if pw.cell_parameters is not None:
        cp = pw.cell_parameters
        block = [f"CELL_PARAMETERS {{{cp.option}}}"]
        for v in cp.vectors:
            block.append(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}")
        sections.append("\n".join(block))

    return "\n\n".join(sections) + "\n"


def save_pw_input(pw: PWInput, path: str | Path) -> None:
    """Write a *PWInput* to a file on disk.

    Parameters
    ----------
    pw : PWInput
        The input object to serialize.
    path : str or Path
        Destination file path.
    """
    Path(path).write_text(write_pw_input(pw))
