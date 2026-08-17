"""Outer-loop convergence evaluator and DFT/correlated-solver coordinator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from qeanalyzer.io.pw_input import PWInput
from qeanalyzer.models import QERunResult
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult
from qeanalyzer.workflow.decisions import NextRunDecision

if TYPE_CHECKING:
    from qeanalyzer.quantum.feedback import QuantumFeedbackPolicy


@dataclass
class ConvergenceCriteria:
    energy_tolerance_ev: float = 1e-4
    rdm_tolerance: float = 1e-3
    gradient_tolerance: float = 1e-3
    max_outer_iterations: int = 15
    require_rdm: bool = True
    require_gradient: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy_tolerance_ev": self.energy_tolerance_ev,
            "rdm_tolerance": self.rdm_tolerance,
            "gradient_tolerance": self.gradient_tolerance,
            "max_outer_iterations": self.max_outer_iterations,
            "require_rdm": self.require_rdm,
            "require_gradient": self.require_gradient,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvergenceCriteria":
        return cls(
            energy_tolerance_ev=data.get("energy_tolerance_ev", 1e-4),
            rdm_tolerance=data.get("rdm_tolerance", 1e-3),
            gradient_tolerance=data.get("gradient_tolerance", 1e-3),
            max_outer_iterations=data.get("max_outer_iterations", 15),
            require_rdm=data.get("require_rdm", True),
            require_gradient=data.get("require_gradient", True),
        )


@dataclass
class ConvergenceCheckResult:
    is_converged: bool
    iteration: int
    delta_energy_ev: float | None = None
    delta_rdm_frobenius: float | None = None
    max_gradient: float | None = None
    passed_criteria: dict[str, bool | None] = field(default_factory=dict)
    reason: str = ""

    def summary(self) -> str:
        lines = [
            f"Outer-Loop Status: {'CONVERGED' if self.is_converged else 'IN_PROGRESS'} (Iteration {self.iteration})",
            "=" * 50,
        ]
        values = (
            ("energy", "|ΔE|", self.delta_energy_ev, "eV"),
            ("rdm", "||Δγ||_F", self.delta_rdm_frobenius, ""),
            ("gradient", "max |g_k|", self.max_gradient, ""),
        )
        for key, label, value, unit in values:
            state = self.passed_criteria.get(key)
            status = "N/A" if state is None else ("PASS" if state else "FAIL")
            rendered = "not available" if value is None else f"{abs(value):.6e} {unit}".rstrip()
            lines.append(f"{label:<16}: {rendered} [{status}]")
        lines.append(f"Reason           : {self.reason}")
        return "\n".join(lines)


@dataclass
class OuterLoopIterationRecord:
    iteration_index: int
    dft_run_id: str
    dft_energy_ev: float
    quantum_energy_ev: float
    total_energy_ev: float
    one_rdm: list[list[float]] = field(default_factory=list)
    natural_occupations: list[float] = field(default_factory=list)
    adapt_operators: list[str] = field(default_factory=list)
    max_gradient: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_index": self.iteration_index,
            "dft_run_id": self.dft_run_id,
            "dft_energy_ev": self.dft_energy_ev,
            "quantum_energy_ev": self.quantum_energy_ev,
            "total_energy_ev": self.total_energy_ev,
            "one_rdm": self.one_rdm,
            "natural_occupations": list(self.natural_occupations),
            "adapt_operators": list(self.adapt_operators),
            "max_gradient": self.max_gradient,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OuterLoopIterationRecord":
        return cls(
            iteration_index=data["iteration_index"],
            dft_run_id=data["dft_run_id"],
            dft_energy_ev=data["dft_energy_ev"],
            quantum_energy_ev=data["quantum_energy_ev"],
            total_energy_ev=data["total_energy_ev"],
            one_rdm=data.get("one_rdm", []),
            natural_occupations=data.get("natural_occupations", []),
            adapt_operators=data.get("adapt_operators", []),
            max_gradient=data.get("max_gradient"),
            metadata=data.get("metadata", {}),
        )


class OuterLoopLedger:
    def __init__(self, criteria: ConvergenceCriteria | None = None) -> None:
        self.criteria = criteria or ConvergenceCriteria()
        self.iterations: list[OuterLoopIterationRecord] = []

    def record_iteration(self, dft_result: QERunResult, quantum_result: QuantumRunResult,
                         metadata: dict[str, Any] | None = None) -> OuterLoopIterationRecord:
        idx = len(self.iterations) + 1
        dft_energy = dft_result.electronic.total_energy_ev
        residual = quantum_result.metadata.get("residual_gradient")
        if residual is None and quantum_result.operator_gradients:
            residual = quantum_result.operator_gradients[-1]
        record = OuterLoopIterationRecord(
            iteration_index=idx,
            dft_run_id=dft_result.run_id or f"dft_{idx:03d}",
            dft_energy_ev=round(float(dft_energy or 0.0), 8),
            quantum_energy_ev=round(float(quantum_result.energy_ev), 8),
            total_energy_ev=round(float(quantum_result.energy_ev), 8),
            one_rdm=quantum_result.one_rdm,
            natural_occupations=quantum_result.natural_occupations,
            adapt_operators=quantum_result.selected_operators,
            max_gradient=(None if residual is None else round(float(residual), 10)),
            metadata={
                "quantum_solver_type": quantum_result.solver_type,
                "quantum_converged": quantum_result.converged,
                **(metadata or {}),
            },
        )
        self.iterations.append(record)
        return record

    @staticmethod
    def _rdm_distance(current: list[list[float]], previous: list[list[float]]) -> float | None:
        if not current or not previous or len(current) != len(previous):
            return None
        if any(len(current[i]) != len(previous[i]) for i in range(len(current))):
            return None
        total = 0.0
        for i in range(len(current)):
            for j in range(len(current[i])):
                delta = current[i][j] - previous[i][j]
                total += delta * delta
        return math.sqrt(total)

    def check_convergence(self) -> ConvergenceCheckResult:
        n = len(self.iterations)
        if n == 0:
            return ConvergenceCheckResult(False, 0, reason="No outer iterations recorded yet")
        current = self.iterations[-1]
        if n == 1:
            pass_grad = (
                current.max_gradient is not None
                and current.max_gradient < self.criteria.gradient_tolerance
            ) if self.criteria.require_gradient else None
            return ConvergenceCheckResult(
                False,
                1,
                max_gradient=current.max_gradient,
                passed_criteria={"energy": None, "rdm": None, "gradient": pass_grad},
                reason=(
                    f"Reached maximum allowed outer iterations ({self.criteria.max_outer_iterations})"
                    if n >= self.criteria.max_outer_iterations else
                    "First outer iteration completed; at least two states are needed for ΔE/ΔRDM"
                ),
            )

        previous = self.iterations[-2]
        delta_e = current.total_energy_ev - previous.total_energy_ev
        pass_e = abs(delta_e) < self.criteria.energy_tolerance_ev

        delta_rdm = self._rdm_distance(current.one_rdm, previous.one_rdm)
        if delta_rdm is None:
            pass_rdm: bool | None = False if self.criteria.require_rdm else None
        else:
            pass_rdm = delta_rdm < self.criteria.rdm_tolerance

        if current.max_gradient is None:
            pass_gradient: bool | None = False if self.criteria.require_gradient else None
        else:
            pass_gradient = current.max_gradient < self.criteria.gradient_tolerance

        effective_rdm = True if pass_rdm is None else pass_rdm
        effective_gradient = True if pass_gradient is None else pass_gradient
        converged = pass_e and effective_rdm and effective_gradient
        passed = {"energy": pass_e, "rdm": pass_rdm, "gradient": pass_gradient}

        if converged:
            reason = f"Outer-loop criteria satisfied in {n} iterations"
        elif n >= self.criteria.max_outer_iterations:
            reason = f"Reached maximum allowed outer iterations ({self.criteria.max_outer_iterations})"
        else:
            missing = [key for key, val in passed.items() if val is False]
            reason = "Outer loop not converged; pending/unavailable criteria: " + ", ".join(missing)

        return ConvergenceCheckResult(
            is_converged=converged,
            iteration=n,
            delta_energy_ev=round(delta_e, 10),
            delta_rdm_frobenius=(None if delta_rdm is None else round(delta_rdm, 10)),
            max_gradient=current.max_gradient,
            passed_criteria=passed,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "criteria": self.criteria.to_dict(),
            "iterations": [row.to_dict() for row in self.iterations],
            "convergence": self.check_convergence().summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OuterLoopLedger":
        ledger = cls(ConvergenceCriteria.from_dict(data.get("criteria", {})))
        ledger.iterations = [OuterLoopIterationRecord.from_dict(row) for row in data.get("iterations", [])]
        return ledger


class OuterLoopController:
    def __init__(self, ledger: OuterLoopLedger | None = None,
                 criteria: ConvergenceCriteria | None = None,
                 feedback_policy: str | "QuantumFeedbackPolicy" = "occupation_feedback") -> None:
        self.ledger = ledger or OuterLoopLedger(criteria=criteria)
        self.feedback_policy = feedback_policy

    def step(self, dft_result: QERunResult, quantum_result: QuantumRunResult,
             prev_input: PWInput | None = None,
             workflow_state: dict[str, Any] | None = None) -> NextRunDecision:
        self.ledger.record_iteration(dft_result, quantum_result)
        status = self.ledger.check_convergence()
        if status.is_converged:
            return NextRunDecision(
                decision_type="STOP",
                policy_name="outer_loop_controller",
                policy_version="2.0",
                reason=f"DFT-correlated outer loop converged: {status.reason}",
                target_calculation="none",
                parent_run=dft_result.run_id,
            )
        if len(self.ledger.iterations) >= self.ledger.criteria.max_outer_iterations:
            return NextRunDecision(
                decision_type="STOP",
                policy_name="outer_loop_controller",
                policy_version="2.0",
                reason=status.reason,
                target_calculation="none",
                parent_run=dft_result.run_id,
            )

        if isinstance(self.feedback_policy, str):
            from qeanalyzer.quantum.feedback import apply_quantum_feedback
            decision = apply_quantum_feedback(
                dft_result, quantum_result, policy_name=self.feedback_policy, prev_input=prev_input
            )
        else:
            decision = self.feedback_policy.evaluate_feedback(
                dft_result, quantum_result, prev_input=prev_input, workflow_state=workflow_state
            )
        decision.reason = f"[Outer Iteration {len(self.ledger.iterations)}] {decision.reason}"
        return decision
