#!/usr/bin/env python3
"""Example 4: Multi-step self-consistent DFT-ADAPT outer loop with multi-criteria convergence."""

from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum import (
    ADAPTVQESolver,
    build_active_space_hamiltonian,
    select_active_space,
)
from qeanalyzer.workflow import (
    ConvergenceCriteria,
    OuterLoopController,
    OuterLoopLedger,
)

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"

def main() -> None:
    print("=" * 60)
    print(" Example 4: DFT-ADAPT Outer Loop Orchestration")
    print("=" * 60)

    # 1. Setup outer-loop ledger and convergence criteria
    criteria = ConvergenceCriteria(
        energy_tolerance_ev=1e-4,
        rdm_tolerance=1e-3,
        gradient_tolerance=1e-3,
        max_outer_iterations=5,
    )
    ledger = OuterLoopLedger(criteria=criteria)
    controller = OuterLoopController(ledger=ledger, feedback_policy="occupation_feedback")

    pw_in = read_pw_input(FIXTURES / "si_scf.in")
    pw_out = read_pw_output(FIXTURES / "si_scf.out")
    qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")
    dft_result = build_run_result(pw_in=pw_in, pw_out=pw_out, qe_xml=qe_xml, run_id="dft_demo")

    # 2. Select active space & build Hamiltonian
    asp = select_active_space(dft_result, method="band_index", band_start=1, band_end=2)
    ham = build_active_space_hamiltonian(dft_result, active_space=asp, onsite_u_ev=2.0)

    # 3. Simulate outer-loop steps with ADAPT-VQE
    vqe_solver = ADAPTVQESolver(gradient_threshold=1e-3, max_adapt_iterations=5)

    print("\nStarting Outer-Loop Iterations...")
    for iter_idx in range(1, 4):
        # Solve quantum Hamiltonian
        q_result = vqe_solver.solve(ham, active_space=asp)

        # Coordinate step in controller
        decision = controller.step(dft_result, q_result, prev_input=pw_in)

        conv = ledger.check_convergence()
        print(f"\n[Iteration {iter_idx}] Decision: {decision.decision_type} | Converged: {conv.is_converged}")
        print(f"Reason: {decision.reason}")

        if decision.decision_type == "STOP":
            print("\nOuter loop terminated successfully.")
            break

    # 4. Display ledger summary
    print("\n--- Outer Loop Ledger Summary ---")
    for it in ledger.iterations:
        print(f"Iter {it.iteration_index}: DFT E={it.dft_energy_ev:.4f} eV, Quantum E={it.quantum_energy_ev:.4f} eV, Max Grad={it.max_gradient}")


if __name__ == "__main__":
    main()
