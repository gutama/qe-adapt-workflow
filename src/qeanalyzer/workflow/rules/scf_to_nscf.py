"""Transition policy: Converged SCF -> Non-Self-Consistent Field (NSCF)."""

from __future__ import annotations

import math
from copy import deepcopy

from qeanalyzer.io.pw_input import KPoints, PWInput, write_pw_input
from qeanalyzer.models import QERunResult
from qeanalyzer.workflow.decisions import NextRunDecision
from qeanalyzer.workflow.policy import Policy


class SCFToNSCFPolicy(Policy):
    """Transition from a converged SCF calculation to an NSCF calculation.

    Increases number of bands (nbnd) to include unoccupied states for DOS/Bands
    or active-space orbital selection, and refines k-point grid.
    """

    name: str = "scf_to_nscf"
    version: str = "1.0"
    description: str = "Transition converged SCF calculation to NSCF with expanded empty bands and refined k-mesh"

    def can_handle(self, result: QERunResult, prev_input: PWInput | None = None) -> bool:
        """Check if calculation is a converged SCF run."""
        return result.status.scf_converged and result.calculation == "scf"

    def evaluate(self, result: QERunResult, prev_input: PWInput | None = None) -> NextRunDecision:
        """Construct the next NSCF input."""
        if not self.can_handle(result, prev_input):
            raise ValueError(f"Policy {self.name} cannot handle the provided run result.")

        # Determine target number of bands
        n_elec = result.electronic.n_electrons or 8.0
        min_occupied = math.ceil(n_elec / 2.0)
        curr_bands = result.electronic.n_bands or min_occupied
        target_bands = max(curr_bands + 4, math.ceil(min_occupied * 1.5))

        modified_namelists: dict[str, dict] = {
            "CONTROL": {"calculation": "nscf"},
            "SYSTEM": {"nbnd": target_bands, "nosym": False},
            "ELECTRONS": {"diagonalization": "david", "diago_full_acc": True},
        }

        next_inp = None
        next_text = None
        if prev_input is not None:
            next_inp = deepcopy(prev_input)
            for nl, params in modified_namelists.items():
                for k, v in params.items():
                    next_inp.set_param(nl, k, v)

            # Refine k-points mesh if automatic
            if next_inp.kpoints and next_inp.kpoints.option == "automatic" and next_inp.kpoints.grid:
                gx, gy, gz = next_inp.kpoints.grid
                next_inp.kpoints = KPoints(
                    option="automatic",
                    grid=(max(gx, 6), max(gy, 6), max(gz, 6)),
                    offsets=next_inp.kpoints.offsets,
                )
            next_text = write_pw_input(next_inp)

        return NextRunDecision(
            decision_type="NEXT_RUN",
            policy_name=self.name,
            policy_version=self.version,
            reason=f"SCF converged. Advancing to NSCF with nbnd={target_bands} for electronic DOS/bands analysis.",
            target_calculation="nscf",
            is_restart=False,
            parent_run=result.run_id,
            modified_namelists=modified_namelists,
            next_input=next_inp,
            next_input_text=next_text,
        )
