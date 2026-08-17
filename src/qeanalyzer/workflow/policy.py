"""Policy engine interface and registry for deterministic workflow progression."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qeanalyzer.io.pw_input import PWInput
from qeanalyzer.models import QERunResult
from qeanalyzer.workflow.decisions import NextRunDecision


class Policy(ABC):
    """Abstract base class for calculation transition and recovery policies."""

    name: str = "base_policy"
    version: str = "1.0"
    description: str = "Base policy"

    @abstractmethod
    def can_handle(self, result: QERunResult, prev_input: PWInput | None = None) -> bool:
        """Check if this policy is applicable to the given run result."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, result: QERunResult, prev_input: PWInput | None = None) -> NextRunDecision:
        """Evaluate the run result and generate a NextRunDecision."""
        raise NotImplementedError


class PolicyRegistry:
    """Registry maintaining available workflow policies with priority selection."""

    def __init__(self) -> None:
        self._policies: dict[str, Policy] = {}
        self._ordered_list: list[Policy] = []

    def register(self, policy: Policy) -> None:
        """Register a policy."""
        self._policies[policy.name] = policy
        if policy not in self._ordered_list:
            self._ordered_list.append(policy)

    def get(self, name: str) -> Policy | None:
        """Get policy by name."""
        return self._policies.get(name)

    def select_policy(self, result: QERunResult, prev_input: PWInput | None = None) -> Policy | None:
        """Find the first matching policy capable of handling the calculation state."""
        for policy in self._ordered_list:
            if policy.can_handle(result, prev_input):
                return policy
        return None

    def list_policies(self) -> list[str]:
        """List all registered policy names."""
        return [p.name for p in self._ordered_list]


# Global default registry
default_registry = PolicyRegistry()
