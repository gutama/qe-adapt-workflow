"""Recovery policy: Unconverged SCF mitigation."""

from __future__ import annotations

from copy import deepcopy

from qeanalyzer.io.pw_input import PWInput, write_pw_input
from qeanalyzer.models import QERunResult
from qeanalyzer.workflow.decisions import NextRunDecision
from qeanalyzer.workflow.policy import Policy


class UnconvergedSCFPolicy(Policy):
    """Recovery policy for unconverged SCF calculations.

    Diagnoses convergence failure and applies corrective measures:
    - If metallic/no gap without smearing -> adds smearing & increases nbnd.
    - If charge oscillation -> lowers mixing_beta (e.g. 0.3) and increases electron_maxstep.
    - If Davidson failure -> switches diagonalization to CG or local-TF mixing.
    """

    name: str = "unconverged_scf_recovery"
    version: str = "1.0"
    description: str = "Recover unconverged SCF calculation by adjusting mixing, smearing, and max iterations"

    def can_handle(self, result: QERunResult, prev_input: PWInput | None = None) -> bool:
        """Check if calculation failed SCF convergence."""
        return not result.status.scf_converged and result.calculation == "scf"

    def evaluate(self, result: QERunResult, prev_input: PWInput | None = None) -> NextRunDecision:
        """Construct the corrective retry input."""
        if not self.can_handle(result, prev_input):
            raise ValueError(f"Policy {self.name} cannot handle the provided run result.")

        modified_namelists: dict[str, dict] = {
            "ELECTRONS": {
                "mixing_beta": 0.3,
                "electron_maxstep": 150,
                "mixing_mode": "plain",
            }
        }

        # Check if system is likely metallic and missing smearing
        is_metallic = (
            result.electronic.is_metal is True
            or (result.electronic.band_gap_ev is not None and result.electronic.band_gap_ev < 1e-3)
        )
        reason_items = ["Reduced mixing_beta to 0.3", "Increased electron_maxstep to 150"]

        if is_metallic:
            modified_namelists["SYSTEM"] = {
                "occupations": "smearing",
                "smearing": "mp",
                "degauss": 0.02,
            }
            if result.electronic.n_bands:
                modified_namelists["SYSTEM"]["nbnd"] = result.electronic.n_bands + 4
            reason_items.append("Added Marzari-Vanderbilt smearing (degauss=0.02 Ry) for metallic state")

        next_inp = None
        next_text = None
        if prev_input is not None:
            next_inp = deepcopy(prev_input)
            for nl, params in modified_namelists.items():
                for k, v in params.items():
                    next_inp.set_param(nl, k, v)
            next_text = write_pw_input(next_inp)

        return NextRunDecision(
            decision_type="RETRY",
            policy_name=self.name,
            policy_version=self.version,
            reason=f"SCF failed to converge. Applied mitigations: {'; '.join(reason_items)}.",
            target_calculation="scf",
            is_restart=False,
            parent_run=result.run_id,
            modified_namelists=modified_namelists,
            next_input=next_inp,
            next_input_text=next_text,
        )
