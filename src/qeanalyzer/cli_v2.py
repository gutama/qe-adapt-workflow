"""Hardened CLI facade around the original command implementations."""

from __future__ import annotations

import sys
from typing import Sequence

from qeanalyzer import cli as _legacy
from qeanalyzer.io.source_bundle import detect_and_load_sources
from qeanalyzer.quantum.hamiltonian import build_band_model_hamiltonian

_original_export = _legacy.cmd_export_fcidump


def _export_band_model(args):
    sys.stderr.write(
        "WARNING: export-fcidump currently exports a parameterized QE band-derived model, "
        "not an ab-initio QE Hamiltonian. For physical materials integrals use "
        "QE -> Wannier/downfolding/cRPA -> build_integral_hamiltonian -> FCIDUMP.\n"
    )
    return _original_export(args)


def _install_hardened_boundaries() -> None:
    _legacy._detect_and_load_sources = detect_and_load_sources
    _legacy.build_active_space_hamiltonian = build_band_model_hamiltonian
    _legacy.cmd_export_fcidump = _export_band_model


def main(argv: Sequence[str] | None = None) -> int:
    _install_hardened_boundaries()
    return _legacy.main(list(argv) if argv is not None else None)
