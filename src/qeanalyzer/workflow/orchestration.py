"""Workflow orchestration and next calculation directory staging."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from qeanalyzer.io.pw_input import PWInput, write_pw_input
from qeanalyzer.models import QERunResult
from qeanalyzer.workflow.decisions import NextRunDecision
from qeanalyzer.workflow.ledger import WorkflowLedger, WorkflowRunEntry
from qeanalyzer.workflow.policy import default_registry


def plan_next_calculation(
    result: QERunResult,
    prev_input: PWInput | None = None,
    policy_name: str | None = None,
    output_dir: str | Path | None = None,
    ledger: WorkflowLedger | None = None,
    next_run_id: str | None = None,
    input_filename: str = "pw.in",
) -> NextRunDecision:
    """Evaluate current calculation result, plan next step, and optionally stage directory.

    Parameters
    ----------
    result : QERunResult
        Current run result.
    prev_input : PWInput, optional
        Previous input parameters.
    policy_name : str, optional
        Specific policy name to apply. If None, auto-selects from registry.
    output_dir : str or Path, optional
        Destination directory to stage the next calculation input.
    ledger : WorkflowLedger, optional
        Ledger to record the new run entry in.
    next_run_id : str, optional
        Unique ID for the new calculation run.
    input_filename : str, optional
        Name of generated input file (default: 'pw.in').

    Returns
    -------
    NextRunDecision
        The evaluated workflow decision.
    """
    policy = None
    if policy_name:
        policy = default_registry.get(policy_name)
        if policy is None:
            raise ValueError(f"Unknown workflow policy: '{policy_name}'")
    else:
        policy = default_registry.select_policy(result, prev_input=prev_input)

    if policy is None:
        return NextRunDecision(
            decision_type="STOP",
            policy_name="none",
            reason="No applicable policy rule matched calculation state.",
            target_calculation=result.calculation,
            parent_run=result.run_id,
        )

    decision = policy.evaluate(result, prev_input=prev_input)

    # If staging directory requested
    if output_dir is not None and decision.next_input is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        in_file = out_p / input_filename
        content = decision.next_input_text or write_pw_input(decision.next_input)
        in_file.write_text(content, encoding="utf-8")

        # Record in ledger if available
        if ledger is not None:
            new_id = next_run_id or f"{decision.target_calculation}-{len(ledger.runs) + 1:03d}"
            entry = WorkflowRunEntry(
                run_id=new_id,
                calculation=decision.target_calculation,
                parent_run=decision.parent_run,
                status="planned",
                run_dir=str(out_p),
                input_file=str(in_file),
                timestamp=datetime.now(timezone.utc).isoformat(),
                policy_applied=decision.policy_name,
            )
            ledger.add_run(entry)

    return decision
