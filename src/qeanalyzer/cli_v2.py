"""Backwards-compatible console entry point.

The run-coherence and band-model boundaries this module used to install by
rebinding attributes on :mod:`qeanalyzer.cli` now live in the library itself, so
every caller gets them -- including code that imports :mod:`qeanalyzer.cli`
directly.  This module remains only so the ``qeanalyzer`` console script and any
existing ``qeanalyzer.cli_v2`` imports keep working.
"""

from __future__ import annotations

from typing import Sequence

from qeanalyzer.cli import main as _main

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the single CLI implementation."""
    return _main(list(argv) if argv is not None else None)
