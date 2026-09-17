#!/usr/bin/env python3
"""Example 5: several correlated methods on one Hamiltonian, all recorded.

The workflow is not ADAPT-only.  Any method that implements ``QuantumSolver``
can be named in the registry, and ``compare_solvers`` runs a list of them
against *one* Hamiltonian so their answers are comparable at all.

What this example does not do is declare a winner on cost.  Arms here differ in
budget on purpose -- a small A-CASE basis against a larger one -- and the
comparison record says the budget was not matched, because matching is a claim
about how the arms were configured and only the caller can make it.

The A-CASE arms need the optional ``clifford_qc`` sibling; without it the
example still runs and says which arms were skipped.
"""

import warnings

from qeanalyzer.quantum import (
    build_hubbard_hamiltonian,
    clifford_qc_subspace_available,
    compare_solvers,
    describe_solvers,
)


def hubbard_chain(n_orbitals: int, n_electrons: float, u: float = 3.0):
    hopping = {}
    for i in range(n_orbitals - 1):
        hopping[(i, i + 1)] = 1.0
        hopping[(i + 1, i)] = 1.0
    return build_hubbard_hamiltonian(
        n_orbitals=n_orbitals, n_electrons=n_electrons, hopping_t=hopping, onsite_u=u
    )


def main() -> None:
    print("Methods this workflow can drive:\n")
    print(describe_solvers())

    # A 4-site Hubbard chain at half filling: small enough for an exact
    # reference, correlated enough that a truncated subspace is visibly above it.
    model = hubbard_chain(4, 4.0)

    arms = [{"solver_type": "exact", "label": "exact_fci"}]
    if clifford_qc_subspace_available():
        arms += [
            {"solver_type": "acase", "label": "acase_m6",
             "options": {"max_basis_size": 6}},
            {"solver_type": "acase", "label": "acase_m12",
             "options": {"max_basis_size": 12}},
            {"solver_type": "adapt_acase", "label": "adapt2_then_acase_m12",
             "options": {"adapt_max_operators": 2, "max_basis_size": 12}},
            {"solver_type": "adapt_vqe", "label": "adapt_8ops",
             "options": {"max_adapt_iterations": 8}},
        ]
        note = ("retained basis size 6 vs 12, ADAPT capped at 8 operators -- "
                "chosen for runtime, NOT matched on cost")
    else:
        note = "exact arm only: clifford_qc is not installed in this environment"

    print("\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        comparison = compare_solvers(model, arms, budget_note=note, on_error="record")
    print(comparison.summary())

    selected = comparison.selected
    print(f"\nLowest energy among the ranked arms: {selected.label} "
          f"at {selected.energy_ev:.8f} eV")
    for arm in comparison.arms:
        if arm.result is None or arm.label == selected.label:
            continue
        gap = comparison.error_vs_reference_ev(arm.label)  # positive: above exact
        residual = arm.result.residual
        print(
            f"  {arm.label:<22} {arm.energy_ev:12.8f} eV  "
            f"above exact by {gap:.6f} eV  "
            f"residual={'n/a' if residual is None else f'{residual:.3e}'} "
            f"({arm.result.residual_kind or 'not reported'})"
        )

    if not clifford_qc_subspace_available():
        print("\nNOTE: install clifford_qc to run the ADAPT-VQE and A-CASE arms.")


if __name__ == "__main__":
    main()
