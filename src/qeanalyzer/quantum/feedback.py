"""Quantum feedback policies translating quantum solver results into next DFT calculations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from qeanalyzer.io.pw_input import PWInput, write_pw_input
from qeanalyzer.models import QERunResult
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult
from qeanalyzer.workflow.decisions import NextRunDecision


class QuantumFeedbackPolicy(ABC):
    """Abstract base class for quantum solver feedback policies.

    Translates correlated many-body outputs (1-RDMs, natural occupations,
    active-space energies) into deterministic modifications for subsequent DFT steps.
    """

    name: str = "base_quantum_feedback"
    version: str = "1.0"
    description: str = "Base quantum feedback policy"

    @abstractmethod
    def evaluate_feedback(
        self,
        qe_result: QERunResult,
        quantum_result: QuantumRunResult,
        prev_input: PWInput | None = None,
        workflow_state: dict[str, Any] | None = None,
    ) -> NextRunDecision:
        """Evaluate quantum result and generate next DFT input decision."""
        ...


class OccupationFeedbackPolicy(QuantumFeedbackPolicy):
    """Adjusts DFT smearing/occupations and electronic mixing based on quantum natural orbital occupations.

    Parameters
    ----------
    correlation_threshold : float, optional
        Fractional occupation threshold indicating multireference character (default: 0.05).
    target_smearing : str, optional
        Smearing type to use when correlated fractional occupations are found (default: 'cold').
    adjusted_degauss : float, optional
        Broadening parameter degauss in Ry (default: 0.01 Ry).
    """

    name: str = "occupation_feedback"
    version: str = "1.0"
    description: str = "Adjusts DFT electronic occupations and mixing from quantum 1-RDM natural occupations"

    def __init__(
        self,
        correlation_threshold: float = 0.05,
        target_smearing: str = "cold",
        adjusted_degauss: float = 0.01,
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.target_smearing = target_smearing
        self.adjusted_degauss = adjusted_degauss

    def evaluate_feedback(
        self,
        qe_result: QERunResult,
        quantum_result: QuantumRunResult,
        prev_input: PWInput | None = None,
        workflow_state: dict[str, Any] | None = None,
    ) -> NextRunDecision:
        nat_occs = quantum_result.natural_occupations
        has_multireference = any(
            self.correlation_threshold < occ < (2.0 - self.correlation_threshold)
            for occ in nat_occs
        )

        modified_namelists: dict[str, dict[str, Any]] = {
            "CONTROL": {"calculation": "scf"},
            "ELECTRONS": {
                "mixing_beta": 0.3,  # Conservative mixing for correlated reoccupation
                "conv_thr": 1.0e-8,
            },
        }

        if has_multireference:
            modified_namelists["SYSTEM"] = {
                "occupations": "smearing",
                "smearing": self.target_smearing,
                "degauss": self.adjusted_degauss,
            }
            reason = (
                f"Detected correlated natural occupations {nat_occs[:4]}. "
                f"Switching to occupations='smearing' (degauss={self.adjusted_degauss} Ry) and damped mixing."
            )
        else:
            reason = f"Natural occupations are single-reference {nat_occs[:4]}. Advancing standard SCF."

        next_inp = None
        next_text = None
        if prev_input is not None:
            next_inp = deepcopy(prev_input)
            for nl, params in modified_namelists.items():
                for k, v in params.items():
                    next_inp.set_param(nl, k, v)
            next_text = write_pw_input(next_inp)

        return NextRunDecision(
            decision_type="NEXT_RUN",
            policy_name=self.name,
            policy_version=self.version,
            reason=reason,
            target_calculation="scf",
            is_restart=False,
            parent_run=qe_result.run_id,
            modified_namelists=modified_namelists,
            next_input=next_inp,
            next_input_text=next_text,
        )


class ActiveSpaceFeedbackPolicy(QuantumFeedbackPolicy):
    """Refines active space size and band count based on quantum correlation energy and boundary occupations.

    Parameters
    ----------
    boundary_occupation_threshold : float, optional
        Threshold for boundary orbital entanglement (default: 0.02).
    band_increment : int, optional
        Number of bands to add when active space expansion is triggered (default: 2).
    """

    name: str = "active_space_feedback"
    version: str = "1.0"
    description: str = "Expands DFT band count and active space window if boundary orbitals remain correlated"

    def __init__(
        self,
        boundary_occupation_threshold: float = 0.02,
        band_increment: int = 2,
    ) -> None:
        self.boundary_occupation_threshold = boundary_occupation_threshold
        self.band_increment = band_increment

    def evaluate_feedback(
        self,
        qe_result: QERunResult,
        quantum_result: QuantumRunResult,
        prev_input: PWInput | None = None,
        workflow_state: dict[str, Any] | None = None,
    ) -> NextRunDecision:
        nat_occs = quantum_result.natural_occupations
        needs_expansion = False

        if nat_occs:
            # Check lowest occupied in active space
            lowest_occ = nat_occs[-1]
            if lowest_occ > self.boundary_occupation_threshold:
                needs_expansion = True

        curr_nbnd = qe_result.electronic.n_bands or 8
        target_nbnd = curr_nbnd + self.band_increment if needs_expansion else curr_nbnd

        modified_namelists: dict[str, dict[str, Any]] = {
            "CONTROL": {"calculation": "scf"},
            "SYSTEM": {"nbnd": target_nbnd},
        }

        if needs_expansion:
            reason = (
                f"Active space boundary orbital has occupation {lowest_occ:.3f} > {self.boundary_occupation_threshold}. "
                f"Expanding active space bands to nbnd={target_nbnd}."
            )
        else:
            reason = "Active space boundary is well-isolated. Maintaining current orbital partition."

        next_inp = None
        next_text = None
        if prev_input is not None:
            next_inp = deepcopy(prev_input)
            for nl, params in modified_namelists.items():
                for k, v in params.items():
                    next_inp.set_param(nl, k, v)
            next_text = write_pw_input(next_inp)

        return NextRunDecision(
            decision_type="NEXT_RUN",
            policy_name=self.name,
            policy_version=self.version,
            reason=reason,
            target_calculation="scf",
            is_restart=False,
            parent_run=qe_result.run_id,
            modified_namelists=modified_namelists,
            next_input=next_inp,
            next_input_text=next_text,
        )


class HubbardUFeedbackPolicy(QuantumFeedbackPolicy):
    """Updates DFT+U Hubbard parameters based on correlated 1-RDM site charge redistribution.

    Parameters
    ----------
    response_alpha : float, optional
        Linear response scaling parameter dU/dn (eV/electron, default: 0.5).
    species_index : int, optional
        Species index for Hubbard_U(i) in Quantum ESPRESSO (default: 1).
    """

    name: str = "hubbard_u_feedback"
    version: str = "1.0"
    description: str = "Adjusts DFT+U Hubbard_U parameter based on quantum 1-RDM local charge redistribution"

    def __init__(
        self,
        response_alpha: float = 0.5,
        species_index: int = 1,
    ) -> None:
        self.response_alpha = response_alpha
        self.species_index = species_index

    def evaluate_feedback(
        self,
        qe_result: QERunResult,
        quantum_result: QuantumRunResult,
        prev_input: PWInput | None = None,
        workflow_state: dict[str, Any] | None = None,
    ) -> NextRunDecision:
        # Determine average charge on site 0 from 1-RDM diagonal
        local_charge = quantum_result.one_rdm[0][0] if quantum_result.one_rdm else 1.0
        # Baseline reference charge
        nominal_charge = quantum_result.n_electrons / max(1, quantum_result.n_orbitals)
        delta_n = local_charge - nominal_charge

        # Updated Hubbard U in eV
        current_u = 0.0
        if prev_input is not None:
            current_u = float(prev_input.get_param("SYSTEM", f"Hubbard_U({self.species_index})", 0.0))

        updated_u = max(0.0, current_u + self.response_alpha * delta_n)

        modified_namelists: dict[str, dict[str, Any]] = {
            "CONTROL": {"calculation": "scf"},
            "SYSTEM": {
                "lda_plus_u": True,
                f"Hubbard_U({self.species_index})": round(updated_u, 4),
            },
        }

        reason = (
            f"Quantum charge fluctuation delta_n={delta_n:.3f}. "
            f"Updating Hubbard_U({self.species_index}) to {updated_u:.4f} eV."
        )

        next_inp = None
        next_text = None
        if prev_input is not None:
            next_inp = deepcopy(prev_input)
            for nl, params in modified_namelists.items():
                for k, v in params.items():
                    next_inp.set_param(nl, k, v)
            next_text = write_pw_input(next_inp)

        return NextRunDecision(
            decision_type="NEXT_RUN",
            policy_name=self.name,
            policy_version=self.version,
            reason=reason,
            target_calculation="scf",
            is_restart=False,
            parent_run=qe_result.run_id,
            modified_namelists=modified_namelists,
            next_input=next_inp,
            next_input_text=next_text,
        )


def apply_quantum_feedback(
    qe_result: QERunResult,
    quantum_result: QuantumRunResult,
    policy_name: str = "occupation_feedback",
    prev_input: PWInput | None = None,
    **kwargs: Any,
) -> NextRunDecision:
    """Convenience entry point to evaluate a quantum feedback policy."""
    p_name = policy_name.lower()
    if p_name in ("occupation_feedback", "occupation", "occ"):
        policy = OccupationFeedbackPolicy(**kwargs)
    elif p_name in ("active_space_feedback", "active_space", "as"):
        policy = ActiveSpaceFeedbackPolicy(**kwargs)
    elif p_name in ("hubbard_u_feedback", "hubbard_u", "hubbard", "u"):
        policy = HubbardUFeedbackPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown quantum feedback policy: '{policy_name}'.")

    return policy.evaluate_feedback(qe_result=qe_result, quantum_result=quantum_result, prev_input=prev_input)
