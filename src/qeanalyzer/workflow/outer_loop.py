"""Outer-loop convergence evaluator and self-consistent DFT-ADAPT coordinator."""

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
    """Thresholds for declaring self-consistent DFT-ADAPT outer-loop convergence."""

    energy_tolerance_ev: float = 1e-4
    rdm_tolerance: float = 1e-3
    gradient_tolerance: float = 1e-3
    max_outer_iterations: int = 15

    def to_dict(self) -> dict[str, Any]:
        """Convert criteria to dictionary."""
        return {
            "energy_tolerance_ev": self.energy_tolerance_ev,
            "rdm_tolerance": self.rdm_tolerance,
            "gradient_tolerance": self.gradient_tolerance,
            "max_outer_iterations": self.max_outer_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceCriteria:
        """Construct criteria from dictionary."""
        return cls(
            energy_tolerance_ev=data.get("energy_tolerance_ev", 1e-4),
            rdm_tolerance=data.get("rdm_tolerance", 1e-3),
            gradient_tolerance=data.get("gradient_tolerance", 1e-3),
            max_outer_iterations=data.get("max_outer_iterations", 15),
        )


@dataclass
class ConvergenceCheckResult:
    """Detailed outcome of an outer-loop convergence assessment."""

    is_converged: bool
    iteration: int
    delta_energy_ev: float | None = None
    delta_rdm_frobenius: float | None = None
    max_gradient: float | None = None
    passed_criteria: dict[str, bool] = field(default_factory=dict)
    reason: str = ""

    def summary(self) -> str:
        """Human-readable summary of convergence status."""
        status_str = "CONVERGED" if self.is_converged else "IN_PROGRESS"
        lines = [
            f"Outer-Loop Status: {status_str} (Iteration {self.iteration})",
            "=" * 50,
        ]
        if self.delta_energy_ev is not None:
            e_pass = self.passed_criteria.get("energy", False)
            lines.append(f"|ΔE|             : {abs(self.delta_energy_ev):.6e} eV [{'PASS' if e_pass else 'FAIL'}]")
        if self.delta_rdm_frobenius is not None:
            rdm_pass = self.passed_criteria.get("rdm", False)
            lines.append(f"||Δγ||_F         : {self.delta_rdm_frobenius:.6e} [{'PASS' if rdm_pass else 'FAIL'}]")
        if self.max_gradient is not None:
            g_pass = self.passed_criteria.get("gradient", False)
            lines.append(f"max |g_k|        : {self.max_gradient:.6e} [{'PASS' if g_pass else 'FAIL'}]")
        lines.append(f"Reason           : {self.reason}")
        return "\n".join(lines)


@dataclass
class OuterLoopIterationRecord:
    """Record of a single DFT-ADAPT outer iteration."""

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
        """Convert record to dictionary."""
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
    def from_dict(cls, data: dict[str, Any]) -> OuterLoopIterationRecord:
        """Construct record from dictionary."""
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
    """Maintains the chronological history of DFT-ADAPT outer iterations and evaluates convergence."""

    def __init__(self, criteria: ConvergenceCriteria | None = None) -> None:
        self.criteria = criteria or ConvergenceCriteria()
        self.iterations: list[OuterLoopIterationRecord] = []

    def record_iteration(
        self,
        dft_result: QERunResult,
        quantum_result: QuantumRunResult,
        metadata: dict[str, Any] | None = None,
    ) -> OuterLoopIterationRecord:
        """Record an outer iteration from DFT and Quantum calculation results."""
        idx = len(self.iterations) + 1
        dft_e = dft_result.electronic.total_energy_ev or 0.0
        q_e = quantum_result.energy_ev
        total_e = q_e  # Total correlated ground-state energy

        # Residual commutator gradient after ADAPT-VQE iterations
        residual_grad = quantum_result.operator_gradients[-1] if quantum_result.operator_gradients else 0.0

        rec = OuterLoopIterationRecord(
            iteration_index=idx,
            dft_run_id=dft_result.run_id or f"dft_{idx:03d}",
            dft_energy_ev=round(dft_e, 8),
            quantum_energy_ev=round(q_e, 8),
            total_energy_ev=round(total_e, 8),
            one_rdm=quantum_result.one_rdm,
            natural_occupations=quantum_result.natural_occupations,
            adapt_operators=quantum_result.selected_operators,
            max_gradient=round(residual_grad, 8) if residual_grad is not None else None,
            metadata=metadata or {},
        )
        self.iterations.append(rec)
        return rec

    def check_convergence(self) -> ConvergenceCheckResult:
        """Evaluate multi-criteria convergence against the recorded iteration history."""
        n_iters = len(self.iterations)
        if n_iters == 0:
            return ConvergenceCheckResult(
                is_converged=False,
                iteration=0,
                reason="No outer iterations recorded yet.",
            )

        curr = self.iterations[-1]

        # The iteration limit is evaluated only after the convergence criteria
        # below: a loop that satisfies every criterion on its final allowed
        # iteration has converged, and must not be reported as having run out.

        # Single iteration cannot compute delta_E or delta_rdm
        if n_iters == 1:
            if n_iters >= self.criteria.max_outer_iterations:
                return ConvergenceCheckResult(
                    is_converged=False,
                    iteration=n_iters,
                    max_gradient=curr.max_gradient,
                    reason=(
                        "Reached maximum allowed outer iterations "
                        f"({self.criteria.max_outer_iterations})."
                    ),
                )
            return ConvergenceCheckResult(
                is_converged=False,
                iteration=1,
                max_gradient=curr.max_gradient,
                reason="First outer iteration completed. Proceeding with self-consistent cycle.",
            )

        prev = self.iterations[-2]

        # 1. Delta Energy |E_n - E_{n-1}|
        delta_e = curr.total_energy_ev - prev.total_energy_ev
        pass_e = abs(delta_e) < self.criteria.energy_tolerance_ev

        # 2. Frobenius norm difference of 1-RDM ||gamma_n - gamma_{n-1}||_F
        delta_rdm = 0.0
        pass_rdm = True
        if curr.one_rdm and prev.one_rdm and len(curr.one_rdm) == len(prev.one_rdm):
            norb = len(curr.one_rdm)
            sq_diff = 0.0
            for p in range(norb):
                for q in range(norb):
                    diff = curr.one_rdm[p][q] - prev.one_rdm[p][q]
                    sq_diff += diff * diff
            delta_rdm = math.sqrt(sq_diff)
            pass_rdm = delta_rdm < self.criteria.rdm_tolerance

        # 3. Maximum commutator gradient
        pass_grad = True
        if curr.max_gradient is not None:
            pass_grad = curr.max_gradient < self.criteria.gradient_tolerance

        is_conv = pass_e and pass_rdm and pass_grad
        passed_map = {
            "energy": pass_e,
            "rdm": pass_rdm,
            "gradient": pass_grad,
        }

        if is_conv:
            reason = (
                f"Outer-loop self-consistency achieved in {n_iters} iterations "
                f"(|ΔE|={abs(delta_e):.2e} eV, ||Δγ||={delta_rdm:.2e})."
            )
        elif n_iters >= self.criteria.max_outer_iterations:
            reason = (
                "Reached maximum allowed outer iterations "
                f"({self.criteria.max_outer_iterations})."
            )
        else:
            failed = [k for k, v in passed_map.items() if not v]
            reason = f"Outer loop not converged. Pending criteria: {', '.join(failed)}."

        return ConvergenceCheckResult(
            is_converged=is_conv,
            iteration=n_iters,
            delta_energy_ev=round(delta_e, 8),
            delta_rdm_frobenius=round(delta_rdm, 8),
            max_gradient=curr.max_gradient,
            passed_criteria=passed_map,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize outer ledger to dictionary."""
        return {
            "criteria": self.criteria.to_dict(),
            "iterations": [it.to_dict() for it in self.iterations],
            "convergence": self.check_convergence().summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OuterLoopLedger:
        """Construct ledger from dictionary."""
        crit = ConvergenceCriteria.from_dict(data.get("criteria", {}))
        ledger = cls(criteria=crit)
        for it_data in data.get("iterations", []):
            ledger.iterations.append(OuterLoopIterationRecord.from_dict(it_data))
        return ledger


class OuterLoopController:
    """Coordinates outer-loop step progression, feedback application, and termination."""

    def __init__(
        self,
        ledger: OuterLoopLedger | None = None,
        criteria: ConvergenceCriteria | None = None,
        feedback_policy: str | QuantumFeedbackPolicy = "occupation_feedback",
    ) -> None:
        self.ledger = ledger or OuterLoopLedger(criteria=criteria)
        self.feedback_policy = feedback_policy

    def step(
        self,
        dft_result: QERunResult,
        quantum_result: QuantumRunResult,
        prev_input: PWInput | None = None,
        workflow_state: dict[str, Any] | None = None,
    ) -> NextRunDecision:
        """Evaluate current step, record history, check convergence, and generate next decision."""
        self.ledger.record_iteration(dft_result=dft_result, quantum_result=quantum_result)
        status = self.ledger.check_convergence()

        if status.is_converged:
            return NextRunDecision(
                decision_type="STOP",
                policy_name="outer_loop_controller",
                policy_version="1.0",
                reason=f"DFT-ADAPT Outer Loop Converged: {status.reason}",
                target_calculation="none",
                is_restart=False,
                parent_run=dft_result.run_id,
            )

        if len(self.ledger.iterations) >= self.ledger.criteria.max_outer_iterations:
            return NextRunDecision(
                decision_type="STOP",
                policy_name="outer_loop_controller",
                policy_version="1.0",
                reason=f"DFT-ADAPT Outer Loop Stopped: Reached maximum iterations ({self.ledger.criteria.max_outer_iterations}).",
                target_calculation="none",
                is_restart=False,
                parent_run=dft_result.run_id,
            )

        # Apply quantum feedback to generate next calculation input
        if isinstance(self.feedback_policy, str):
            from qeanalyzer.quantum.feedback import apply_quantum_feedback
            decision = apply_quantum_feedback(
                qe_result=dft_result,
                quantum_result=quantum_result,
                policy_name=self.feedback_policy,
                prev_input=prev_input,
            )
        else:
            decision = self.feedback_policy.evaluate_feedback(
                qe_result=dft_result,
                quantum_result=quantum_result,
                prev_input=prev_input,
                workflow_state=workflow_state,
            )

        decision.reason = f"[Outer Iteration {len(self.ledger.iterations)}] {decision.reason}"
        return decision
