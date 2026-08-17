"""Transition policy: Converged Relax/VC-Relax -> Static SCF on Optimized Structure."""

from __future__ import annotations

from copy import deepcopy

from qeanalyzer.io.pw_input import AtomicPosition, CellParameters, PWInput, write_pw_input
from qeanalyzer.models import QERunResult
from qeanalyzer.workflow.decisions import NextRunDecision
from qeanalyzer.workflow.policy import Policy


class RelaxToSCFPolicy(Policy):
    """Transition from a converged ionic or variable-cell relaxation to static SCF.

    Extracts final relaxed coordinates (and cell vectors if vc-relax) and
    initializes a high-precision static ground-state calculation.
    """

    name: str = "relax_to_scf"
    version: str = "1.0"
    description: str = "Extract relaxed geometry and transition to high-precision static SCF"

    def can_handle(self, result: QERunResult, prev_input: PWInput | None = None) -> bool:
        """Check if calculation is a converged relax or vc-relax calculation.

        Only geometry convergence qualifies.  Electronic (SCF) convergence of the
        final ionic step says nothing about whether BFGS reached its force and
        energy thresholds, so it must not stand in for ``opt_converged``.
        """
        is_relax_calc = result.calculation in ("relax", "vc-relax")
        return is_relax_calc and result.status.opt_converged is True

    def evaluate(self, result: QERunResult, prev_input: PWInput | None = None) -> NextRunDecision:
        """Construct the static SCF input with relaxed geometry."""
        if not self.can_handle(result, prev_input):
            raise ValueError(f"Policy {self.name} cannot handle the provided run result.")

        modified_namelists: dict[str, dict] = {
            "CONTROL": {"calculation": "scf"},
            "ELECTRONS": {"conv_thr": 1.0e-9},
        }

        next_inp = None
        next_text = None
        if prev_input is not None:
            next_inp = deepcopy(prev_input)
            for nl, params in modified_namelists.items():
                for k, v in params.items():
                    next_inp.set_param(nl, k, v)

            # Update structure from final structure if available
            if result.structure and result.structure.atoms:
                # Update positions
                new_positions: list[AtomicPosition] = []
                for atom in result.structure.atoms:
                    new_positions.append(AtomicPosition(
                        symbol=atom.symbol,
                        position=atom.position_bohr,
                    ))
                next_inp.atomic_positions = new_positions
                next_inp.atomic_positions_option = "bohr"

            # Update cell parameters if vc-relax
            if (
                result.calculation == "vc-relax"
                and result.structure
                and result.structure.cell_vectors_bohr
            ):
                next_inp.cell_parameters = CellParameters(
                    vectors=result.structure.cell_vectors_bohr,
                    option="bohr",
                )
                next_inp.set_param("SYSTEM", "ibrav", 0)

            next_text = write_pw_input(next_inp)

        return NextRunDecision(
            decision_type="NEXT_RUN",
            policy_name=self.name,
            policy_version=self.version,
            reason="Geometry optimization converged. Transitioning to static SCF on the relaxed structure.",
            target_calculation="scf",
            is_restart=False,
            parent_run=result.run_id,
            modified_namelists=modified_namelists,
            next_input=next_inp,
            next_input_text=next_text,
        )
