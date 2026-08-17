"""Recovery policy: Interrupted calculation restart."""

from __future__ import annotations

from copy import deepcopy

from qeanalyzer.io.pw_input import PWInput, write_pw_input
from qeanalyzer.models import QERunResult
from qeanalyzer.workflow.decisions import NextRunDecision
from qeanalyzer.workflow.policy import Policy


class InterruptedRecoveryPolicy(Policy):
    """Recovery policy for abruptly interrupted calculations (walltime, node failure).

    Enforces RESTART semantics: sets restart_mode = 'restart' and continues
    the calculation in-place from stored wavefunctions and charge density.
    """

    name: str = "interrupted_calculation_restart"
    version: str = "1.0"
    description: str = "Restart interrupted calculation from checkpointed save directory"

    def can_handle(self, result: QERunResult, prev_input: PWInput | None = None) -> bool:
        """Check if calculation was interrupted before normal completion.

        A hard QE error (``exit_status == 'error'``) also leaves the run
        incomplete, but restarting it unchanged just reproduces the failure.
        Those are excluded so the corrective policies can claim them instead.
        """
        if result.status.exit_status == "error":
            return False
        return (
            not result.status.completed
            or result.status.exit_status == "interrupted"
        )

    def evaluate(self, result: QERunResult, prev_input: PWInput | None = None) -> NextRunDecision:
        """Construct the restart input."""
        if not self.can_handle(result, prev_input):
            raise ValueError(f"Policy {self.name} cannot handle the provided run result.")

        modified_namelists: dict[str, dict] = {
            "CONTROL": {"restart_mode": "restart"}
        }

        next_inp = None
        next_text = None
        if prev_input is not None:
            next_inp = deepcopy(prev_input)
            for nl, params in modified_namelists.items():
                for k, v in params.items():
                    next_inp.set_param(nl, k, v)
            next_text = write_pw_input(next_inp)

        return NextRunDecision(
            decision_type="RESTART",
            policy_name=self.name,
            policy_version=self.version,
            reason="Calculation was terminated before completion. Setting restart_mode='restart'.",
            target_calculation=result.calculation or "scf",
            is_restart=True,  # RESTART semantics
            parent_run=result.run_id,
            modified_namelists=modified_namelists,
            next_input=next_inp,
            next_input_text=next_text,
        )
