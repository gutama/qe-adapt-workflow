"""FCIDUMP format reader and deterministic exporter for electronic Hamiltonians."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from qeanalyzer.quantum.hamiltonian import MaterialHamiltonian


def _format_fortran_value(val: Any) -> str:
    """Format Python values into Fortran namelist literal syntax."""
    if isinstance(val, bool):
        return ".TRUE." if val else ".FALSE."
    if isinstance(val, (list, tuple)):
        return ",".join(str(x) for x in val)
    return str(val)


def write_fcidump(
    hamiltonian: MaterialHamiltonian,
    path: str | Path | None = None,
    orbsym: list[int] | None = None,
    isym: int = 1,
    tolerance: float = 1e-12,
) -> str:
    """Export MaterialHamiltonian to standard electronic FCIDUMP format.

    Parameters
    ----------
    hamiltonian : MaterialHamiltonian
        The electronic Hamiltonian containing 1-body and 2-body integrals.
    path : str or Path, optional
        Destination file path to save FCIDUMP output.
    orbsym : list of int, optional
        Orbital point-group symmetry irreps (default: all 1).
    isym : int, optional
        Total wave function symmetry irrep (default: 1).
    tolerance : float, optional
        Threshold below which integrals are omitted as zero (default: 1e-12).

    Returns
    -------
    str
        Formatted FCIDUMP text.
    """
    norb = hamiltonian.n_orbitals
    nelec = int(round(hamiltonian.n_electrons))
    ms2 = hamiltonian.spin
    sym_list = orbsym if orbsym is not None else [1] * norb

    # 1. Header namelist &FCI
    lines = [
        "&FCI",
        f" NORB={norb},",
        f" NELEC={nelec},",
        f" MS2={ms2},",
        f" ORBSYM={','.join(str(s) for s in sym_list)},",
        f" ISYM={isym},",
        "/",
    ]

    # 2. 2-Body Integrals in Chemists' notation (ij|kl)
    # Canonical index ordering: i >= j, k >= l, and pair(i, j) >= pair(k, l)
    # pair index: p_idx(i, j) = i * (i + 1) // 2 + j
    for i in range(1, norb + 1):
        for j in range(1, i + 1):
            ij_pair = i * (i + 1) // 2 + j
            for k in range(1, norb + 1):
                for l in range(1, k + 1):
                    kl_pair = k * (k + 1) // 2 + l
                    if ij_pair < kl_pair:
                        continue

                    # Retrieve (ij|kl) in chemists' notation: h2[i-1][j-1][k-1][l-1]
                    val = hamiltonian.h2[i - 1][j - 1][k - 1][l - 1]
                    if abs(val) > tolerance:
                        lines.append(f"{val:23.16E} {i:4d} {j:4d} {k:4d} {l:4d}")

    # 3. 1-Body Integrals h_ij (i >= j, k=0, l=0)
    for i in range(1, norb + 1):
        for j in range(1, i + 1):
            val = hamiltonian.h1[i - 1][j - 1]
            if abs(val) > tolerance:
                lines.append(f"{val:23.16E} {i:4d} {j:4d} {0:4d} {0:4d}")

    # 4. Constant energy shift E_const (i=0, j=0, k=0, l=0)
    const_val = hamiltonian.constant
    lines.append(f"{const_val:23.16E} {0:4d} {0:4d} {0:4d} {0:4d}")

    text = "\n".join(lines) + "\n"

    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    return text


def parse_fcidump(text: str) -> MaterialHamiltonian:
    """Parse standard FCIDUMP content string into a MaterialHamiltonian.

    Parameters
    ----------
    text : str
        FCIDUMP file content.

    Returns
    -------
    MaterialHamiltonian
        Reconstructed Hamiltonian with 1-body and 2-body integrals.
    """
    lines = text.strip().splitlines()
    if not lines:
        raise ValueError("Empty FCIDUMP content.")

    # 1. Parse header namelist
    header_lines: list[str] = []
    body_start_idx = 0

    for idx, line in enumerate(lines):
        header_lines.append(line)
        if "/" in line or "&END" in line.upper():
            body_start_idx = idx + 1
            break

    header_text = " ".join(header_lines)

    # Extract NORB, NELEC, MS2
    norb_match = re.search(r"NORB\s*=\s*(\d+)", header_text, re.IGNORECASE)
    if not norb_match:
        raise ValueError("NORB parameter missing in FCIDUMP header.")
    norb = int(norb_match.group(1))

    nelec_match = re.search(r"NELEC\s*=\s*(\d+)", header_text, re.IGNORECASE)
    nelec = int(nelec_match.group(1)) if nelec_match else 0

    ms2_match = re.search(r"MS2\s*=\s*(-?\d+)", header_text, re.IGNORECASE)
    ms2 = int(ms2_match.group(1)) if ms2_match else 0

    # Initialize integral tensors
    h1 = [[0.0 for _ in range(norb)] for _ in range(norb)]
    h2 = [[[[0.0 for _ in range(norb)] for _ in range(norb)] for _ in range(norb)] for _ in range(norb)]
    constant = 0.0

    # 2. Parse body integral lines
    for line in lines[body_start_idx:]:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            continue

        try:
            val = float(parts[0])
            i = int(parts[1])
            j = int(parts[2])
            k = int(parts[3])
            l = int(parts[4])
        except ValueError:
            continue

        if i == 0 and j == 0 and k == 0 and l == 0:
            # Constant energy shift
            constant = val
        elif k == 0 and l == 0:
            # 1-body integral h_{ij} = h_{ji}
            p = i - 1
            q = j - 1
            if 0 <= p < norb and 0 <= q < norb:
                h1[p][q] = val
                h1[q][p] = val
        else:
            # 2-body integral (ij|kl) in chemists' notation
            # Map into h2[p][r][q][s] = (pr|qs) = (ij|kl)
            # Permutations for 8-fold real symmetry:
            # (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
            perms = [
                (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i),
            ]
            for p1, p2, p3, p4 in perms:
                # In chemist notation: (p1 p2 | p3 p4) -> in h2 tensor h2[p1-1][p3-1][p2-1][p4-1]
                idx1 = p1 - 1
                idx2 = p2 - 1
                idx3 = p3 - 1
                idx4 = p4 - 1
                if 0 <= idx1 < norb and 0 <= idx2 < norb and 0 <= idx3 < norb and 0 <= idx4 < norb:
                    h2[idx1][idx2][idx3][idx4] = val

    return MaterialHamiltonian(
        n_orbitals=norb,
        n_electrons=float(nelec),
        constant=constant,
        spin=ms2,
        energy_unit="eV",
        h1=h1,
        h2=h2,
        metadata={"source": "fcidump"},
    )


def read_fcidump(path: str | Path) -> MaterialHamiltonian:
    """Read an FCIDUMP file from disk and return a MaterialHamiltonian.

    Parameters
    ----------
    path : str or Path
        Path to the FCIDUMP file.

    Returns
    -------
    MaterialHamiltonian
        Parsed Hamiltonian.
    """
    text = Path(path).read_text(encoding="utf-8")
    return parse_fcidump(text)
