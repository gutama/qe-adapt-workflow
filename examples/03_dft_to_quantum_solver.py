#!/usr/bin/env python3
"""Example 3: QE band diagnostics -> explicit heuristic model -> quantum solver.

This example deliberately does *not* claim that Kohn-Sham bands are an
ab-initio FCIDUMP.  The physical QE->many-body path requires localized/downfolded
one- and two-electron integrals (e.g. Wannier90 + screened interactions).
"""

import tempfile
from pathlib import Path

from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.quantum import (
    ADAPTVQESolver,
    ExactDiagonalizationSolver,
    build_band_model_hamiltonian,
    clifford_qc_available,
    read_fcidump,
    select_active_space,
    write_fcidump,
)

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"


def main() -> None:
    pw_in = read_pw_input(FIXTURES / "si_scf.in")
    result = build_run_result(
        pw_in=pw_in,
        pw_out=read_pw_output(FIXTURES / "si_scf.out"),
        qe_xml=read_qe_xml(FIXTURES / "si_scf.xml"),
        run_id="si_scf",
    )
    active = select_active_space(result, method="band_index", band_start=1, band_end=2)
    model = build_band_model_hamiltonian(
        result, active_space=active, onsite_u_ev=2.5, intersite_v_ev=0.5
    )
    print(model.summary())
    print("Model status:", model.metadata["scientific_status"])
    print("Warning:", model.metadata["warning"])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "heuristic_model.FCIDUMP"
        write_fcidump(model, path=path)
        loaded = read_fcidump(path)
        exact = ExactDiagonalizationSolver().solve(loaded)
        print(f"Exact reference for this *heuristic model*: {exact.energy_ev:.8f} eV")

        if clifford_qc_available():
            adapt = ADAPTVQESolver(
                gradient_threshold=1e-6,
                max_adapt_iterations=10,
                compute_exact_reference=True,
            ).solve(loaded)
            print(f"clifford_qc ADAPT-VQE: {adapt.energy_ev:.8f} eV")
            print("Residual commutator gradient:", adapt.metadata["residual_gradient"])
        else:
            print("Real ADAPT skipped: install sibling ../clifford_qc[openfermion].")


if __name__ == "__main__":
    main()
