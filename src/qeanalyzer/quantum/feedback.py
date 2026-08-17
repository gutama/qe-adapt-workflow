"""Experimental quantum-to-DFT feedback policies.

These policies are controller heuristics, not a derived DFT+many-body
self-consistency functional.  Every decision is marked
``scientific_status=experimental_heuristic`` so reports cannot silently present
it as a validated physical feedback scheme.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from qeanalyzer.io.pw_input import PWInput, write_pw_input
from qeanalyzer.models import QERunResult
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult
from qeanalyzer.workflow.decisions import NextRunDecision

_STATUS = "experimental_heuristic"


class QuantumFeedbackPolicy(ABC):
    name = "base_quantum_feedback"
    version = "2.0"
    scientific_status = _STATUS

    @abstractmethod
    def evaluate_feedback(
        self,
        qe_result: QERunResult,
        quantum_result: QuantumRunResult,
        prev_input: PWInput | None = None,
        workflow_state: dict[str, Any] | None = None,
    ) -> NextRunDecision:
        ...

    def _decision(
        self,
        qe_result: QERunResult,
        reason: str,
        modified_namelists: dict[str, dict[str, Any]],
        prev_input: PWInput | None,
    ) -> NextRunDecision:
        next_input = None
        text = None
        if prev_input is not None:
            next_input = deepcopy(prev_input)
            for namelist, params in modified_namelists.items():
                for key, value in params.items():
                    next_input.set_param(namelist, key, value)
            text = write_pw_input(next_input)
        return NextRunDecision(
            decision_type="NEXT_RUN",
            policy_name=self.name,
            policy_version=self.version,
            reason=f"[EXPERIMENTAL HEURISTIC] {reason}",
            target_calculation="scf",
            is_restart=False,
            parent_run=qe_result.run_id,
            modified_namelists=modified_namelists,
            next_input=next_input,
            next_input_text=text,
            metadata={
                "scientific_status": self.scientific_status,
                "validated_physical_self_consistency": False,
            },
        )


class OccupationFeedbackPolicy(QuantumFeedbackPolicy):
    """Smearing/mixing heuristic triggered by correlated natural occupations.

    Despite the historical class name, this does not inject the many-body
    natural occupations into QE.  It only changes smearing and mixing controls.
    """

    name = "occupation_feedback"

    def __init__(self, correlation_threshold: float = 0.05,
                 target_smearing: str = "cold", adjusted_degauss: float = 0.01) -> None:
        self.correlation_threshold = float(correlation_threshold)
        self.target_smearing = target_smearing
        self.adjusted_degauss = float(adjusted_degauss)

    def evaluate_feedback(self, qe_result: QERunResult, quantum_result: QuantumRunResult,
                          prev_input: PWInput | None = None,
                          workflow_state: dict[str, Any] | None = None) -> NextRunDecision:
        occupations = quantum_result.natural_occupations
        correlated = any(
            self.correlation_threshold < occ < 2.0 - self.correlation_threshold
            for occ in occupations
        )
        changes: dict[str, dict[str, Any]] = {
            "CONTROL": {"calculation": "scf"},
            "ELECTRONS": {"mixing_beta": 0.3, "conv_thr": 1.0e-8},
        }
        if correlated:
            changes["SYSTEM"] = {
                "occupations": "smearing",
                "smearing": self.target_smearing,
                "degauss": self.adjusted_degauss,
            }
            reason = (
                f"Natural occupations show multireference character ({occupations[:4]}). "
                "Use more conservative SCF smearing/mixing; this is not an RDM embedding update."
            )
        else:
            reason = "Natural occupations do not trigger the smearing heuristic; retain standard SCF controls."
        return self._decision(qe_result, reason, changes, prev_input)


class ActiveSpaceFeedbackPolicy(QuantumFeedbackPolicy):
    """Heuristic request for more bands when the current active space looks saturated."""

    name = "active_space_feedback"

    def __init__(self, boundary_occupation_threshold: float = 0.02, band_increment: int = 2) -> None:
        self.boundary_occupation_threshold = float(boundary_occupation_threshold)
        self.band_increment = int(band_increment)

    def evaluate_feedback(self, qe_result: QERunResult, quantum_result: QuantumRunResult,
                          prev_input: PWInput | None = None,
                          workflow_state: dict[str, Any] | None = None) -> NextRunDecision:
        occupations = quantum_result.natural_occupations
        lowest = occupations[-1] if occupations else None
        expand = lowest is not None and lowest > self.boundary_occupation_threshold
        current = qe_result.electronic.n_bands or 8
        target = current + self.band_increment if expand else current
        reason = (
            f"Lowest natural occupation {lowest:.4g} exceeds heuristic threshold; request nbnd={target}."
            if expand else
            "Current natural occupations do not trigger heuristic band-count expansion."
        )
        return self._decision(
            qe_result,
            reason,
            {"CONTROL": {"calculation": "scf"}, "SYSTEM": {"nbnd": target}},
            prev_input,
        )


class HubbardUFeedbackPolicy(QuantumFeedbackPolicy):
    """Legacy linear U-adjustment heuristic.

    The mapping from an active-orbital 1-RDM diagonal to an atomic-species DFT+U
    parameter is not generally physical.  This policy is retained for controller
    experiments and is explicitly marked experimental.  It also targets the
    legacy ``Hubbard_U(i)`` input syntax; production QE >=7.1 workflows should
    use a dedicated HUBBARD-card policy after a validated orbital/species map is
    available.
    """

    name = "hubbard_u_feedback"

    def __init__(self, response_alpha: float = 0.5, species_index: int = 1) -> None:
        self.response_alpha = float(response_alpha)
        self.species_index = int(species_index)

    def evaluate_feedback(self, qe_result: QERunResult, quantum_result: QuantumRunResult,
                          prev_input: PWInput | None = None,
                          workflow_state: dict[str, Any] | None = None) -> NextRunDecision:
        local = quantum_result.one_rdm[0][0] if quantum_result.one_rdm else 1.0
        nominal = quantum_result.n_electrons / max(1, quantum_result.n_orbitals)
        delta = local - nominal
        current = 0.0
        if prev_input is not None:
            current = float(prev_input.get_param("SYSTEM", f"Hubbard_U({self.species_index})", 0.0))
        updated = max(0.0, current + self.response_alpha * delta)
        changes = {
            "CONTROL": {"calculation": "scf"},
            "SYSTEM": {
                "lda_plus_u": True,
                f"Hubbard_U({self.species_index})": round(updated, 4),
            },
        }
        return self._decision(
            qe_result,
            f"Legacy linear response heuristic: delta_n={delta:.4g}, proposed U={updated:.4f} eV.",
            changes,
            prev_input,
        )


def apply_quantum_feedback(
    qe_result: QERunResult,
    quantum_result: QuantumRunResult,
    policy_name: str = "occupation_feedback",
    prev_input: PWInput | None = None,
    **kwargs: Any,
) -> NextRunDecision:
    key = policy_name.lower()
    if key in {"occupation_feedback", "occupation", "occ"}:
        policy: QuantumFeedbackPolicy = OccupationFeedbackPolicy(**kwargs)
    elif key in {"active_space_feedback", "active_space", "as"}:
        policy = ActiveSpaceFeedbackPolicy(**kwargs)
    elif key in {"hubbard_u_feedback", "hubbard_u", "hubbard", "u"}:
        policy = HubbardUFeedbackPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown quantum feedback policy {policy_name!r}")
    return policy.evaluate_feedback(qe_result, quantum_result, prev_input=prev_input)
