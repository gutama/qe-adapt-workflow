"""Run several correlated methods on one Hamiltonian and keep every arm.

A comparison is a *record*, not a verdict.  Two methods run on the same active
space can be compared as variational upper bounds, but "which is better" also
depends on a budget -- basis size, operator count, shots -- that this module
cannot infer.  It therefore records what each arm was given, refuses to rank an
arm that is not a scientific result at all, and states plainly that no budget
was matched unless the caller declares one.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from qeanalyzer.quantum.active_space import ActiveSpace
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult, QuantumSolver
from qeanalyzer.quantum.hamiltonian import MaterialHamiltonian

# A comparison ranks variational answers to the same problem.  A fabricated
# trajectory is not one, and neither is another comparison record, so they are
# recorded and skipped instead of quietly winning on energy -- the mock derives
# its numbers from the exact solution and would often "win".
UNRANKABLE_STATUSES = frozenset({"workflow_mock", "comparison_record"})


def hamiltonian_fingerprint(hamiltonian: MaterialHamiltonian) -> str:
    """SHA-256 of the canonical Hamiltonian record.

    Every arm must have been handed the same problem for the comparison to mean
    anything; the digest is what a ledger entry can be checked against later.
    """
    canonical = json.dumps(hamiltonian.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SolverArm:
    """One method, its options, and the label it is reported under."""

    solver_type: str
    label: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.label or self.solver_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.name,
            "solver_type": self.solver_type,
            "options": dict(self.options),
        }


def coerce_arm(arm: Any) -> SolverArm:
    """Accept ``"acase"``, ``("acase", {...})``, a mapping, or a SolverArm."""
    if isinstance(arm, SolverArm):
        return arm
    if isinstance(arm, str):
        return SolverArm(solver_type=arm)
    if isinstance(arm, Mapping):
        data = dict(arm)
        return SolverArm(
            solver_type=data.pop("solver_type"),
            label=data.pop("label", ""),
            options=data.pop("options", data),
        )
    if isinstance(arm, Sequence) and len(arm) == 2:
        solver_type, options = arm
        return SolverArm(solver_type=solver_type, options=dict(options))
    raise TypeError(f"cannot read {arm!r} as a comparison arm")


@dataclass
class ArmOutcome:
    """What one arm produced, including the arms that failed."""

    label: str
    solver_type: str
    options: dict[str, Any] = field(default_factory=dict)
    result: QuantumRunResult | None = None
    error: str | None = None
    wall_seconds: float = 0.0

    @property
    def energy_ev(self) -> float | None:
        return None if self.result is None else self.result.energy_ev

    @property
    def scientific_status(self) -> str:
        if self.result is None:
            return "failed"
        return str(self.result.metadata.get("scientific_status", ""))

    @property
    def rankable(self) -> bool:
        return self.result is not None and self.scientific_status not in UNRANKABLE_STATUSES

    def to_dict(self, include_result: bool = False) -> dict[str, Any]:
        """Compact by default: a ledger wants the numbers, not every RDM."""
        row: dict[str, Any] = {
            "label": self.label,
            "solver_type": self.solver_type,
            "options": dict(self.options),
            "energy_ev": self.energy_ev,
            "converged": None if self.result is None else self.result.converged,
            "residual": None if self.result is None else self.result.residual,
            "residual_kind": None if self.result is None else self.result.residual_kind,
            "scientific_status": self.scientific_status,
            "rankable": self.rankable,
            # Wall time on whatever machine this ran on.  It is provenance, not
            # a hardware cost model, and never a quantum resource count.
            "wall_seconds": round(self.wall_seconds, 6),
            "error": self.error,
        }
        if self.result is not None:
            row["solver_reported_type"] = self.result.solver_type
            row["basis_or_operator_count"] = len(self.result.selected_operators)
        if include_result and self.result is not None:
            row["result"] = self.result.to_dict()
        return row


@dataclass
class SolverComparison:
    """Every arm of one comparison, plus which arm was selected and why."""

    arms: list[ArmOutcome]
    hamiltonian_sha256: str
    n_orbitals: int
    n_electrons: float
    selection_rule: str = "lowest_energy"
    selected_label: str | None = None
    budget_note: str = ""

    @property
    def outcomes(self) -> dict[str, ArmOutcome]:
        return {arm.label: arm for arm in self.arms}

    @property
    def selected(self) -> ArmOutcome | None:
        if self.selected_label is None:
            return None
        return self.outcomes[self.selected_label]

    @property
    def reference_energy_ev(self) -> float | None:
        """Energy of an exact arm, when the comparison included one."""
        for arm in self.arms:
            if arm.result is not None and arm.scientific_status == "exact_reference":
                return arm.energy_ev
        return None

    def error_vs_reference_ev(self, label: str) -> float | None:
        reference = self.reference_energy_ev
        arm = self.outcomes[label]
        if reference is None or arm.energy_ev is None:
            return None
        return arm.energy_ev - reference

    def to_dict(self, include_results: bool = False) -> dict[str, Any]:
        return {
            "scientific_status": "comparison_record",
            "hamiltonian_sha256": self.hamiltonian_sha256,
            "n_orbitals": self.n_orbitals,
            "n_electrons": self.n_electrons,
            "selection_rule": self.selection_rule,
            "selected_label": self.selected_label,
            # Budget matching is a claim about how the arms were configured, so
            # it is only ever what the caller declared -- never inferred here.
            "matched_budget_declared": bool(self.budget_note),
            "budget_note": self.budget_note,
            "reference_energy_ev": self.reference_energy_ev,
            "arms": [
                {
                    **arm.to_dict(include_result=include_results),
                    "error_vs_reference_ev": self.error_vs_reference_ev(arm.label),
                }
                for arm in self.arms
            ],
        }

    def summary(self) -> str:
        reference = self.reference_energy_ev
        # Labels are caller-chosen, so the column is sized to the longest one
        # rather than to a width that silently misaligns the table.
        width = max([len("arm")] + [len(arm.label) for arm in self.arms]) + 2
        rule = width + 60
        lines = [
            f"Solver comparison ({len(self.arms)} arms) on "
            f"{self.n_orbitals} orbitals / {self.n_electrons:g} electrons",
            f"Hamiltonian sha256: {self.hamiltonian_sha256[:16]}...",
            "=" * rule,
            f"{'arm':<{width}}{'energy (eV)':>16}{'vs exact':>14}  {'status':<26}rank",
        ]
        for arm in self.arms:
            if arm.result is None:
                lines.append(f"{arm.label:<{width}}{'FAILED':>16}{'':>14}  "
                             f"{arm.scientific_status:<26}no")
                lines.append(f"{'':<{width}}{arm.error}")
                continue
            delta = self.error_vs_reference_ev(arm.label)
            rendered = "-" if delta is None else f"{delta:+.6e}"
            lines.append(
                f"{arm.label:<{width}}{arm.energy_ev:>16.8f}{rendered:>14}  "
                f"{arm.scientific_status:<26}{'yes' if arm.rankable else 'no'}"
            )
        lines.append("-" * rule)
        lines.append(f"Selected: {self.selected_label or 'none'} (rule: {self.selection_rule})")
        if reference is None:
            lines.append("No exact arm: energies are comparable to each other, not to a reference.")
        lines.append(
            "Budget: " + (self.budget_note if self.budget_note
                          else "NOT DECLARED -- arms were not matched on cost by this runner.")
        )
        return "\n".join(lines)


def compare_solvers(
    hamiltonian: MaterialHamiltonian,
    arms: Iterable[Any],
    active_space: ActiveSpace | None = None,
    selection: str = "lowest_energy",
    budget_note: str = "",
    on_error: str = "raise",
) -> SolverComparison:
    """Run every arm on the same Hamiltonian and record all of them.

    ``selection`` is either ``"lowest_energy"`` -- the variational reading,
    restricted to arms that are scientific results -- or the label of the arm to
    select regardless of energy.  ``on_error="record"`` keeps a failing arm as a
    recorded failure instead of ending the comparison, which is what a long
    outer loop wants when one optional backend is missing.
    """
    from qeanalyzer.quantum.solver_api import create_quantum_solver

    if on_error not in {"raise", "record"}:
        raise ValueError("on_error must be 'raise' or 'record'")
    specs = [coerce_arm(arm) for arm in arms]
    if not specs:
        raise ValueError("a comparison needs at least one arm")
    labels = [spec.name for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"comparison arm labels must be unique, got {labels}")

    outcomes: list[ArmOutcome] = []
    for spec in specs:
        started = time.perf_counter()
        outcome = ArmOutcome(label=spec.name, solver_type=spec.solver_type,
                             options=dict(spec.options))
        try:
            solver = create_quantum_solver(spec.solver_type, **dict(spec.options))
            outcome.result = solver.solve(hamiltonian, active_space=active_space)
        except Exception as exc:  # noqa: BLE001 - recorded verbatim below
            if on_error == "raise":
                raise
            outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.wall_seconds = time.perf_counter() - started
        outcomes.append(outcome)

    # A solver that answered a different particle-number sector did not solve
    # the same problem, and comparing the energies would be meaningless.
    for outcome in outcomes:
        if outcome.result is None:
            continue
        if outcome.result.n_orbitals != hamiltonian.n_orbitals:
            raise ValueError(
                f"arm {outcome.label!r} reported {outcome.result.n_orbitals} orbitals "
                f"for a {hamiltonian.n_orbitals}-orbital Hamiltonian"
            )
        if abs(outcome.result.n_electrons - hamiltonian.n_electrons) > 1e-9:
            raise ValueError(
                f"arm {outcome.label!r} reported {outcome.result.n_electrons} electrons "
                f"for a {hamiltonian.n_electrons:g}-electron Hamiltonian"
            )

    comparison = SolverComparison(
        arms=outcomes,
        hamiltonian_sha256=hamiltonian_fingerprint(hamiltonian),
        n_orbitals=hamiltonian.n_orbitals,
        n_electrons=float(hamiltonian.n_electrons),
        selection_rule=selection,
        budget_note=budget_note,
    )
    comparison.selected_label = _select(comparison, selection)
    return comparison


def _select(comparison: SolverComparison, selection: str) -> str | None:
    if selection == "lowest_energy":
        ranked = [arm for arm in comparison.arms if arm.rankable]
        if not ranked:
            return None
        return min(ranked, key=lambda arm: arm.energy_ev).label
    outcomes = comparison.outcomes
    if selection not in outcomes:
        raise ValueError(
            f"selection {selection!r} is neither 'lowest_energy' nor one of the "
            f"arm labels {sorted(outcomes)}"
        )
    if outcomes[selection].result is None:
        raise ValueError(f"selected arm {selection!r} did not produce a result")
    return selection


class ComparisonSolver(QuantumSolver):
    """A :class:`QuantumSolver` that is really several, run side by side.

    ``solve`` returns the selected arm's own result, with the whole comparison
    attached under ``metadata["comparison"]`` -- so an outer loop keeps driving
    on one energy and one 1-RDM while the ledger still records what every other
    method said about the same Hamiltonian.
    """

    def __init__(
        self,
        arms: Iterable[Any] = ("exact",),
        selection: str = "lowest_energy",
        budget_note: str = "",
        on_error: str = "raise",
        include_arm_results: bool = False,
    ) -> None:
        self.arms = [coerce_arm(arm) for arm in arms]
        self.selection = selection
        self.budget_note = budget_note
        self.on_error = on_error
        self.include_arm_results = bool(include_arm_results)

    def solve(
        self,
        hamiltonian: MaterialHamiltonian,
        active_space: ActiveSpace | None = None,
        initial_state: Any = None,
        **kwargs: Any,
    ) -> QuantumRunResult:
        if initial_state is not None:
            raise NotImplementedError("a comparison does not forward custom initial states")
        comparison = compare_solvers(
            hamiltonian,
            self.arms,
            active_space=active_space,
            selection=self.selection,
            budget_note=self.budget_note,
            on_error=self.on_error,
        )
        selected = comparison.selected
        if selected is None or selected.result is None:
            failures = "; ".join(
                f"{arm.label}: {arm.error}" for arm in comparison.arms if arm.error
            )
            raise RuntimeError(
                "no comparison arm produced a rankable result"
                + (f" ({failures})" if failures else "")
            )
        # A copy: attaching the record to the arm's own result object would
        # nest the comparison inside one of its own entries.
        return replace(
            selected.result,
            metadata={
                **selected.result.metadata,
                "selected_arm": selected.label,
                "selected_from": [arm.label for arm in comparison.arms],
                "comparison": comparison.to_dict(include_results=self.include_arm_results),
            },
        )


__all__ = [
    "ArmOutcome",
    "ComparisonSolver",
    "SolverArm",
    "SolverComparison",
    "UNRANKABLE_STATUSES",
    "coerce_arm",
    "compare_solvers",
    "hamiltonian_fingerprint",
]
