#!/usr/bin/env python3
"""Example 2: Deterministic calculation transitions (Relax -> SCF -> NSCF) and DAG ledger."""

import tempfile
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output
from qeanalyzer.models import build_run_result
from qeanalyzer.workflow import WorkflowLedger, plan_next_calculation

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"

def main() -> None:
    print("=" * 60)
    print(" Example 2: Workflow Progression & DAG Ledger")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ledger_path = tmp_path / "workflow.json"
        ledger = WorkflowLedger(workflow_id="silicon_workflow_demo")

        # Step 1: Process converged relax calculation
        pw_in_relax = read_pw_input(FIXTURES / "si_vc_relax.in")
        pw_out_relax = read_pw_output(FIXTURES / "si_vc_relax.out")
        res_relax = build_run_result(pw_in=pw_in_relax, pw_out=pw_out_relax, run_id="001_relax")

        decision_scf = plan_next_calculation(
            result=res_relax,
            prev_input=pw_in_relax,
            output_dir=tmp_path / "002_scf",
            ledger=ledger,
        )
        print(f"\nStep 1 Decision: {decision_scf.decision_type} (Policy: {decision_scf.policy_name})")
        print(f"Reason         : {decision_scf.reason}")
        print(f"Staged Dir     : {tmp_path / '002_scf'}")

        # Step 2: Process converged SCF calculation
        pw_in_scf = read_pw_input(FIXTURES / "si_scf.in")
        pw_out_scf = read_pw_output(FIXTURES / "si_scf.out")
        res_scf = build_run_result(pw_in=pw_in_scf, pw_out=pw_out_scf, run_id="002_scf")

        decision_nscf = plan_next_calculation(
            result=res_scf,
            prev_input=pw_in_scf,
            output_dir=tmp_path / "003_nscf",
            ledger=ledger,
        )
        print(f"\nStep 2 Decision: {decision_nscf.decision_type} (Policy: {decision_nscf.policy_name})")
        print(f"Reason         : {decision_nscf.reason}")

        # Display ledger summary
        ledger.save_json(ledger_path)
        print(f"\nWorkflow Ledger Saved ({len(ledger.runs)} runs recorded):")
        for r in ledger.runs:
            print(f" - Run: {r.run_id:<12} Calc: {r.calculation:<8} Parent: {r.parent_run or '-':<12} Policy: {r.policy_applied}")


if __name__ == "__main__":
    main()
