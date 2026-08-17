"""Diagnostics and analysis data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DiagnosticLevel = Literal["OK", "INFO", "WARNING", "ERROR"]


@dataclass
class Diagnostic:
    """A diagnostic check result."""

    level: DiagnosticLevel
    category: str
    message: str
    details: str | None = None

    @property
    def symbol(self) -> str:
        """Visual symbol for reports."""
        if self.level == "OK":
            return "✓"
        if self.level == "INFO":
            return "ℹ"
        if self.level == "WARNING":
            return "⚠"
        if self.level == "ERROR":
            return "✗"
        return "•"
