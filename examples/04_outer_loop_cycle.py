#!/usr/bin/env python3
"""Example 4: controller plumbing with an explicitly non-scientific mock.

A physical closed DFT-many-body loop requires a validated feedback functional.
This example tests orchestration only, so both the solver mock and feedback
policy are labelled accordingly and the gradient criterion is disabled.
"""

import warnings
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum import (
    SimulatedADAPTVQESolver,
    build_band_model_hamiltonian,
    select_active_space,
)
from qeanalyzer.workflow import ConvergenceCriteria, OuterLoopController

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"


def main() -> None:
    pw_in = read_pw_input(FIXTURES / "si_scf.in")
    dft = build_run_result(
        pw_in=pw_in,
        pw_out=read_pw_output(FIXTURES / "si_scf.out"),
        qe_xml=read_qe_xml(FIXTURES / "si_scf.xml"),
        run_id="dft_demo",
    )
    active = select_active_space(dft, method="band_index", band_start=1, band_end=2)
    model = build_band_model_hamiltonian(dft, active_space=active, onsite_u_ev=2.0)

    controller = OuterLoopController(
        criteria=ConvergenceCriteria(
            energy_tolerance_ev=1e-4,
            rdm_tolerance=1e-3,
            require_gradient=False,  # mock has no physical commutator residual
            max_outer_iterations=5,
        ),
        feedback_policy="occupation_feedback",  # experimental heuristic
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mock = SimulatedADAPTVQESolver(max_adapt_iterations=8)

    for iteration in range(1, 4):
        qresult = mock.solve(model, active_space=active)
        decision = controller.step(dft, qresult, prev_input=pw_in)
        print(iteration, decision.decision_type, decision.reason)
        if decision.decision_type == "STOP":
            break

    print("NOTE: this was a workflow plumbing demonstration, not physical DFT-ADAPT self-consistency.")


if __name__ == "__main__":
    main()
