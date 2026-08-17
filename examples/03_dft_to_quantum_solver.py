#!/usr/bin/env python3
"""Example 3: DFT active-space selection, Hamiltonian construction, FCIDUMP export, and ADAPT-VQE solver."""

import tempfile
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum import (
    ADAPTVQESolver,
    ExactDiagonalizationSolver,
    apply_quantum_feedback,
    build_active_space_hamiltonian,
    read_fcidump,
    select_active_space,
    write_fcidump,
)

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"

def main() -> None:
    print("=" * 60)
    print(" Example 3: DFT -> Active Space -> ADAPT-VQE -> Feedback")
    print("=" * 60)

    # 1. Parse DFT electronic state
    pw_in = read_pw_input(FIXTURES / "si_scf.in")
    pw_out = read_pw_output(FIXTURES / "si_scf.out")
    qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")
    result = build_run_result(pw_in=pw_in, pw_out=pw_out, qe_xml=qe_xml, run_id="si_scf")

    # 2. Select active space around Fermi energy
    asp = select_active_space(result, method="band_index", band_start=1, band_end=2)
    print(f"\n1. Active Space Selected: {asp.summary()}")

    # 3. Construct parameterized MaterialHamiltonian
    ham = build_active_space_hamiltonian(result, active_space=asp, onsite_u_ev=2.5, intersite_v_ev=0.5)
    print(f"\n2. Material Hamiltonian: {ham.summary()}")

    # 4. Export to FCIDUMP interchange file and roundtrip check
    with tempfile.TemporaryDirectory() as tmpdir:
        fci_path = Path(tmpdir) / "model.fcidump"
        write_fcidump(ham, path=fci_path)
        print(f"\n3. Exported FCIDUMP ({fci_path.stat().st_size} bytes)")
        ham_reloaded = read_fcidump(fci_path)

    # 5. Solve using Exact Diagonalization (FCI) reference
    ed_solver = ExactDiagonalizationSolver()
    ed_result = ed_solver.solve(ham_reloaded, active_space=asp)
    print(f"\n4. Exact Ground State: {ed_result.energy_ev:.6f} eV (Natural Occs: {ed_result.natural_occupations})")

    # 6. Solve using ADAPT-VQE algorithm
    adapt_solver = ADAPTVQESolver(gradient_threshold=1e-3, max_adapt_iterations=10)
    vqe_result = adapt_solver.solve(ham_reloaded, active_space=asp)
    print(f"\n5. ADAPT-VQE Result: {vqe_result.energy_ev:.6f} eV ({len(vqe_result.selected_operators)} operators selected)")

    # 7. Apply quantum feedback policy for next DFT calculation
    decision = apply_quantum_feedback(result, vqe_result, policy_name="occupation", prev_input=pw_in)
    print(f"\n6. Feedback Decision: {decision.decision_type} (Policy: {decision.policy_name})")
    print(f"Reason: {decision.reason}")


if __name__ == "__main__":
    main()
