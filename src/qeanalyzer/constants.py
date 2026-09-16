"""Physical constants shared by QE parsing, the data model and quantum interchange.

One definition per constant.  These values cross two boundaries that have to
agree exactly: the QE XML boundary (Hartree/Ry energies, Bohr coordinates,
Hartree/Bohr^3 stress) and the FCIDUMP boundary (Hartree integrals).  While each
side carried its own copy, a CODATA update applied to one of them would have
made XML-parsed energies and FCIDUMP energies quietly inconsistent.
"""

from __future__ import annotations

HARTREE_TO_RY: float = 2.0
HARTREE_TO_EV: float = 27.211386245988
BOHR_TO_ANGSTROM: float = 0.529177210903
# 1 Hartree / Bohr^3 to kbar: 1 Ha = 4.3597447222071e-18 J, 1 Bohr = 5.29177210903e-11 m
# 1 Ha/Bohr^3 = 2.942102648438959e13 Pa = 2.942102648438959e8 bar = 294210.2648438959 kbar
# Cross-check: QE's uakbar = 147105.0 kbar per Ry/Bohr^3, and 1 Ha = 2 Ry.
HARTREE_BOHR3_TO_KBAR: float = 294210.2648438959

__all__ = [
    "BOHR_TO_ANGSTROM",
    "HARTREE_BOHR3_TO_KBAR",
    "HARTREE_TO_EV",
    "HARTREE_TO_RY",
]
