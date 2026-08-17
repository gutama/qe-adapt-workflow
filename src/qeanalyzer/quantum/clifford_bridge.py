"""Optional real ADAPT-VQE backend implemented by the sibling ``clifford_qc`` project.

``qe-adapt-workflow`` owns QE parsing, workflow state and Hamiltonian provenance.
Quantum-algorithm implementation belongs in ``clifford_qc``; this adapter keeps
that ownership boundary explicit instead of maintaining a second ADAPT engine.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from qeanalyzer.quantum.active_space import ActiveSpace
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult, QuantumSolver
from qeanalyzer.quantum.hamiltonian import MaterialHamiltonian
from qeanalyzer.quantum.units import HARTREE_TO_EV, energy_to_hartree, require_integer_electron_sector


def _imports() -> dict[str, Any]:
    try:
        from clifford_qc.algorithms.adapt import ansatz_program, run_adapt
        from clifford_qc.backends.exact_mv import ExactMVBackend
        from clifford_qc.fermion import c_op, cdag_op
        from clifford_qc.measurement.bank import CommutatorBank
        from clifford_qc.models.chemistry import excitation_pool
        from clifford_qc.models.fcidump import FCIDump, model_from_fcidump
        from clifford_qc.states import expectation
    except ImportError as exc:
        raise ImportError(
            "Real ADAPT-VQE is provided by the sibling clifford_qc project. "
            "Install it in the same environment, e.g. "
            "`pip install -e ../clifford_qc[openfermion]`. "
            "Use SimulatedADAPTVQESolver only for workflow plumbing tests."
        ) from exc
    return locals()


def clifford_qc_available() -> bool:
    try:
        _imports()
    except ImportError:
        return False
    return True


def _to_clifford_fcidump(ham: MaterialHamiltonian) -> Any:
    api = _imports()
    FCIDump = api["FCIDump"]
    nelec = require_integer_electron_sector(ham.n_electrons)
    n = ham.n_orbitals
    one = np.asarray(
        [[energy_to_hartree(ham.h1[p][q], ham.energy_unit) for q in range(n)] for p in range(n)],
        dtype=float,
    )
    two = np.asarray(
        [[[[energy_to_hartree(ham.h2[p][q][r][s], ham.energy_unit)
             for s in range(n)] for r in range(n)] for q in range(n)] for p in range(n)],
        dtype=float,
    )
    core = energy_to_hartree(ham.constant, ham.energy_unit)
    canonical = json.dumps(ham.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return FCIDump(
        n_orbitals=n,
        n_electrons=nelec,
        ms2=int(ham.spin),
        one_body=one,
        two_body=two,
        core_energy=core,
        orbsym=tuple(int(x) for x in ham.metadata.get("orbsym", [1] * n)),
        isym=int(ham.metadata.get("isym", 1)),
        source_path="<qeanalyzer-memory>",
        source_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _spatial_one_rdm(api: dict[str, Any], model: Any, pool: list[Any], result: Any) -> list[list[float]]:
    ansatz_program = api["ansatz_program"]
    ExactMVBackend = api["ExactMVBackend"]
    cdag_op, c_op, expectation = api["cdag_op"], api["c_op"], api["expectation"]
    by_label = {op.label: op for op in pool}
    try:
        chosen = [by_label[label] for label in result.labels]
    except KeyError as exc:
        raise RuntimeError(f"clifford_qc ADAPT result references unknown pool label {exc.args[0]!r}") from exc
    backend = ExactMVBackend()
    rho = backend.state(ansatz_program(model, chosen), result.parameters)
    n_spatial = model.n // 2
    gamma = np.zeros((n_spatial, n_spatial), dtype=float)
    for p in range(n_spatial):
        for q in range(n_spatial):
            value = 0.0
            for spin in (0, 1):
                op = cdag_op(model.n, 2 * p + spin) * c_op(model.n, 2 * q + spin)
                value += float(expectation(rho, op).real)
            gamma[p, q] = value
    # Numerical noise can make the exact real-sector result microscopically asymmetric.
    gamma = 0.5 * (gamma + gamma.T)
    return gamma.tolist()


def _residual_gradient(api: dict[str, Any], model: Any, pool: list[Any], result: Any) -> float:
    ansatz_program = api["ansatz_program"]
    ExactMVBackend = api["ExactMVBackend"]
    CommutatorBank = api["CommutatorBank"]
    selected = set(result.labels)
    candidates = [i for i, op in enumerate(pool) if op.label not in selected]
    if not candidates:
        return 0.0
    by_label = {op.label: op for op in pool}
    chosen = [by_label[label] for label in result.labels]
    rho = ExactMVBackend().state(ansatz_program(model, chosen), result.parameters)
    bank = CommutatorBank(model.hamiltonian, [op.word for op in pool], [op.label for op in pool])
    return max(abs(float(bank.exact_score(i, rho))) for i in candidates)


class CliffordQCADAPTSolver(QuantumSolver):
    """Run the genuine ``clifford_qc.algorithms.adapt.run_adapt`` implementation.

    The default pool is ``clifford_qc.models.chemistry.excitation_pool``: the
    word-level Jordan-Wigner qubit-ADAPT pool derived from particle-number and
    S_z-conserving fermionic singles/doubles.  Energies cross the interface in
    Hartree and are reported here in eV for consistency with ``QuantumRunResult``.
    """

    def __init__(
        self,
        gradient_threshold: float = 1e-6,
        max_adapt_iterations: int = 20,
        optimizer_method: str = "auto",
        maxiter: int = 350,
        compute_exact_reference: bool = False,
    ) -> None:
        self.gradient_threshold = float(gradient_threshold)
        self.max_adapt_iterations = int(max_adapt_iterations)
        self.optimizer_method = optimizer_method
        self.maxiter = int(maxiter)
        self.compute_exact_reference = bool(compute_exact_reference)

    def solve(
        self,
        hamiltonian: MaterialHamiltonian,
        active_space: ActiveSpace | None = None,
        initial_state: Any = None,
        **kwargs: Any,
    ) -> QuantumRunResult:
        if initial_state is not None:
            raise NotImplementedError("custom initial_state handoff to clifford_qc is not implemented yet")
        if not hamiltonian.is_hermitian():
            raise ValueError("ADAPT requires a Hermitian restricted Hamiltonian")
        api = _imports()
        data = _to_clifford_fcidump(hamiltonian)
        model = api["model_from_fcidump"](data, name="qe-adapt-active-space")
        pool = api["excitation_pool"](model.n, data.n_electrons)
        if not pool:
            raise ValueError("clifford_qc excitation pool is empty for this active-space sector")

        result = api["run_adapt"](
            model,
            pool,
            max_operators=self.max_adapt_iterations,
            threshold=self.gradient_threshold,
            optimizer_method=self.optimizer_method,
            maxiter=self.maxiter,
            compute_exact_reference=self.compute_exact_reference,
            track_exact_scores=True,
        )
        one_rdm = _spatial_one_rdm(api, model, pool, result)
        natural = sorted((float(x) for x in np.linalg.eigvalsh(np.asarray(one_rdm))), reverse=True)
        residual = _residual_gradient(api, model, pool, result)

        exact_ha = result.exact_ground_energy
        total_ev = float(result.energy) * HARTREE_TO_EV
        constant_ev = float(data.core_energy) * HARTREE_TO_EV
        hf_ha = api["ExactMVBackend"]().expectation(model.reference, model.hamiltonian, ())
        gradients = [
            float(rec.exact_gradient if rec.exact_gradient is not None else rec.estimate)
            for rec in result.records
            if rec.exact_gradient is not None or rec.estimate is not None
        ]
        iteration_energies = [
            float(rec.energy) * HARTREE_TO_EV for rec in result.records if rec.energy is not None
        ]
        converged = residual < self.gradient_threshold

        return QuantumRunResult(
            energy_ev=round(total_ev, 10),
            electronic_energy_ev=round(total_ev - constant_ev, 10),
            constant_energy_ev=round(constant_ev, 10),
            correlation_energy_ev=round((float(result.energy) - float(hf_ha)) * HARTREE_TO_EV, 10),
            solver_type="clifford_qc_adapt_vqe",
            n_orbitals=hamiltonian.n_orbitals,
            n_electrons=float(data.n_electrons),
            n_spin_orbitals=model.n,
            converged=converged,
            selected_operators=list(result.labels),
            operator_gradients=gradients,
            operator_parameters=[float(x) for x in result.parameters],
            iteration_energies=iteration_energies,
            one_rdm=one_rdm,
            natural_occupations=[round(x, 10) for x in natural],
            metadata={
                "backend_project": "gutama/clifford_qc",
                "backend_api": "clifford_qc.algorithms.adapt.run_adapt",
                "stopped_reason": result.stopped_reason,
                "residual_gradient": residual,
                "exact_ground_energy_ev": (None if exact_ha is None else float(exact_ha) * HARTREE_TO_EV),
                "pool_size": len(pool),
                "energy_interchange_unit": "Hartree",
                "integral_convention": "chemist_(pq|rs)",
            },
        )
