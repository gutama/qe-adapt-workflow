"""Workflow decisions and transition data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from qeanalyzer.io.pw_input import PWInput

DecisionType = Literal["NEXT_RUN", "RESTART", "RETRY", "STOP", "MANUAL_INTERVENTION"]


@dataclass
class NextRunDecision:
    """A deterministic decision defining the next calculation or recovery action."""

    decision_type: DecisionType
    policy_name: str
    policy_version: str = "1.0"
    reason: str = ""
    target_calculation: str = "scf"
    is_restart: bool = False
    parent_run: str | None = None
    modified_namelists: dict[str, dict[str, Any]] = field(default_factory=dict)
    next_input: PWInput | None = None
    next_input_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "policy": {"name": self.policy_name, "version": self.policy_version},
            "reason": self.reason,
            "target_calculation": self.target_calculation,
            "is_restart": self.is_restart,
            "parent_run": self.parent_run,
            "modified_namelists": self.modified_namelists,
            "metadata": dict(self.metadata),
        }
