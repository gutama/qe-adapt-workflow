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
    ExplicitOrbitalSelector,
    build_band_model_hamiltonian,
    build_hubbard_hamiltonian,
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
    # Step 1 -- QE band diagnostics: which bands would carry a correlated model.
    active = select_active_space(result, method="band_index", band_start=1, band_end=2)
    print(active.summary())

    # Step 2 -- choose the model's particle-number sector, explicitly.
    #
    # This SCF record is valence-only: all 4 bands are occupied at every k-point
    # (8 of 8 electrons), so the band-derived sector is a closed shell.  A closed
    # shell has no particle-number- and Sz-conserving excitations, which means
    # ADAPT has an empty operator pool and nothing to grow -- the correlated
    # step cannot say anything about it.
    #
    # The heuristic model therefore fixes its own half-filled sector.  That is a
    # modeling choice about the toy Hamiltonian, *not* the DFT occupation, so it
    # goes through ExplicitOrbitalSelector, which records manual_override=True in
    # the active-space provenance rather than letting the number drift silently.
    model_space = ExplicitOrbitalSelector(
        active_orbitals=active.active_orbitals,
        n_active_electrons=2.0,
    ).select(result)
    print(
        f"Model sector: {model_space.n_active_electrons:g} electrons in "
        f"{model_space.n_active_orbitals} orbitals "
        f"(DFT occupation of these bands: {active.n_active_electrons:g}); "
        f"manual_override={model_space.metadata['manual_override']}"
    )

    # Step 3 -- build the explicitly heuristic model on that sector.
    model = build_band_model_hamiltonian(
        result, active_space=model_space, onsite_u_ev=2.5, intersite_v_ev=0.5
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

        # Step 4 -- why this model is not a quantum-solver problem.
        #
        # build_band_model_hamiltonian is diagonal by construction: k-weighted
        # band energies on the one-body diagonal, density-density U/V on the two-
        # body part, and nothing off-diagonal.  Such a Hamiltonian is diagonal in
        # the occupation-number basis, so every determinant is already an exact
        # eigenstate.  Exact diagonalization simply picks the lowest one, while a
        # variational ansatz started from the aufbau determinant sees a zero
        # gradient for every pool operator and cannot move off it -- it would
        # report that determinant's energy with a residual gradient of 0.0 and
        # look converged while sitting above the ground state.
        #
        # Off-diagonal one-body structure is what makes a model correlated, and
        # deriving it from a QE band structure needs Wannier/downfolding (that is
        # the build_integral_hamiltonian path), which this repository will not
        # fake.  So the hopping below is an explicit model input, not a QE-derived
        # quantity, and the resulting Hamiltonian is a parameterized Hubbard dimer
        # seeded with QE on-site energies -- nothing more.
        # build_hubbard_hamiltonian keeps diagonal entries as on-site energies
        # and negates off-diagonal ones, so a positive amplitude t enters the
        # one-body matrix as the conventional -t.
        hopping_ev = 1.0
        one_body = [list(row) for row in model.h1]
        one_body[0][1] = one_body[1][0] = hopping_ev
        correlated = build_hubbard_hamiltonian(
            n_orbitals=model.n_orbitals,
            n_electrons=model.n_electrons,
            hopping_t=one_body,
            onsite_u=2.5,
            intersite_v={(0, 1): 0.5},
            energy_unit="eV",
        )
        print(
            f"\nCorrelated model: same QE on-site energies plus an explicit "
            f"hopping amplitude t = {hopping_ev:g} eV (a model input, not from QE)"
        )
        correlated_exact = ExactDiagonalizationSolver().solve(correlated)
        print(f"Exact ground state: {correlated_exact.energy_ev:.8f} eV")

        if clifford_qc_available():
            adapt = ADAPTVQESolver(
                gradient_threshold=1e-6,
                max_adapt_iterations=10,
                compute_exact_reference=True,
            ).solve(correlated)
            print(f"clifford_qc ADAPT-VQE: {adapt.energy_ev:.8f} eV")
            print(f"Operators selected: {len(adapt.selected_operators)}")
            print("Residual commutator gradient:", adapt.metadata["residual_gradient"])
            print(f"Error vs exact: {abs(adapt.energy_ev - correlated_exact.energy_ev):.2e} eV")
        else:
            print("Real ADAPT skipped: install sibling ../clifford_qc[openfermion].")


if __name__ == "__main__":
    main()
