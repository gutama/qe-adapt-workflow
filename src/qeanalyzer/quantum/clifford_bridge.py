"""Optional correlated backends implemented by the sibling ``clifford_qc`` project.

``qe-adapt-workflow`` owns QE parsing, workflow state and Hamiltonian provenance.
Quantum-algorithm implementation belongs in ``clifford_qc``; these adapters keep
that ownership boundary explicit instead of maintaining second copies of the
engines here.

Two method families are bridged, and they are different algorithms rather than
two names for one:

* **ADAPT-VQE** (:class:`CliffordQCADAPTSolver`) grows a *unitary ansatz*, one
  rotor at a time, and optimizes its parameters variationally.
* **A-CASE** (:class:`CliffordQCACASESolver`) grows a *linear subspace*
  ``span{A_i|psi>}`` around one reference and solves the projected
  generalized eigenproblem -- Rayleigh-Ritz, with no parameter optimization.
  :class:`CliffordQCADAPTACASESolver` composes the two by handing A-CASE the
  optimized ADAPT state as its reference.

The two loaders are deliberately separate: an older ``clifford_qc`` that
predates the subspace package still drives ADAPT, and only the A-CASE path
reports it missing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from qeanalyzer.quantum.active_space import ActiveSpace
from qeanalyzer.quantum.adapt_bridge import QuantumRunResult, QuantumSolver
from qeanalyzer.quantum.hamiltonian import MaterialHamiltonian
from qeanalyzer.quantum.units import HARTREE_TO_EV, energy_to_hartree, require_integer_electron_sector


@dataclass(frozen=True)
class CliffordQCCoreAPI:
    """The model/state surface every bridged method needs.

    Kept separate from the ADAPT surface because it costs nothing extra: none
    of these need the ``openfermion`` extra, so a method that does not use the
    chemistry excitation pool must not be made to require it.

    Fields keep the upstream spelling so a rename on either side is a normal
    import/attribute error.  The previous ``locals()`` dict resolved the same
    objects by string key, which turned a typo into a KeyError at solve time and
    hid the dependency from every static check.
    """

    ExactMVBackend: Any
    c_op: Any
    cdag_op: Any
    FCIDump: Any
    model_from_fcidump: Any
    expectation: Any


@dataclass(frozen=True)
class CliffordQCAPI(CliffordQCCoreAPI):
    """The core surface plus what ADAPT-VQE adds to it."""

    ansatz_program: Any
    run_adapt: Any
    CommutatorBank: Any
    excitation_pool: Any


@lru_cache(maxsize=1)
def load_clifford_qc_core() -> CliffordQCCoreAPI:
    """Import the backend's model/state surface once and cache the handles."""
    try:
        from clifford_qc.backends.exact_mv import ExactMVBackend
        from clifford_qc.fermion import c_op, cdag_op
        from clifford_qc.models.fcidump import FCIDump, model_from_fcidump
        from clifford_qc.states import expectation
    except ImportError as exc:
        raise ImportError(
            "Correlated solvers are provided by the sibling clifford_qc project. "
            "Install it in the same environment, e.g. "
            "`pip install -e ../clifford_qc[openfermion]`. "
            "Use SimulatedADAPTVQESolver only for workflow plumbing tests."
        ) from exc
    return CliffordQCCoreAPI(
        ExactMVBackend=ExactMVBackend,
        c_op=c_op,
        cdag_op=cdag_op,
        FCIDump=FCIDump,
        model_from_fcidump=model_from_fcidump,
        expectation=expectation,
    )


@lru_cache(maxsize=1)
def load_clifford_qc() -> CliffordQCAPI:
    """Import the optional ADAPT-VQE backend once and cache the handles."""
    core = load_clifford_qc_core()
    try:
        from clifford_qc.algorithms.adapt import ansatz_program, run_adapt
        from clifford_qc.measurement.bank import CommutatorBank
        from clifford_qc.models.chemistry import excitation_pool
    except ImportError as exc:
        raise ImportError(
            "Real ADAPT-VQE is provided by the sibling clifford_qc project, and "
            "its excitation pool needs the chemistry extra. Install it in the "
            "same environment, e.g. `pip install -e ../clifford_qc[openfermion]`. "
            "Use SimulatedADAPTVQESolver only for workflow plumbing tests."
        ) from exc
    return CliffordQCAPI(
        ExactMVBackend=core.ExactMVBackend,
        c_op=core.c_op,
        cdag_op=core.cdag_op,
        FCIDump=core.FCIDump,
        model_from_fcidump=core.model_from_fcidump,
        expectation=core.expectation,
        ansatz_program=ansatz_program,
        run_adapt=run_adapt,
        CommutatorBank=CommutatorBank,
        excitation_pool=excitation_pool,
    )


@dataclass(frozen=True)
class CliffordQCSubspaceAPI:
    """The ``clifford_qc.subspace`` surface the A-CASE adapters use.

    Kept apart from :class:`CliffordQCAPI` so that a sibling checkout without
    the subspace package still supports ADAPT-VQE, and only A-CASE reports the
    missing dependency.
    """

    run_acase: Any
    determinant_excitations: Any
    occupied_spin_orbitals: Any
    dense_residual_norm: Any
    adapt_warm_start: Any


@lru_cache(maxsize=1)
def load_clifford_qc_subspace() -> CliffordQCSubspaceAPI:
    """Import the optional A-CASE surface once and cache the handles."""
    try:
        from clifford_qc.subspace import (
            adapt_warm_start,
            dense_residual_norm,
            determinant_excitations,
            occupied_spin_orbitals,
            run_acase,
        )
    except ImportError as exc:
        raise ImportError(
            "A-CASE is provided by the sibling clifford_qc project "
            "(clifford_qc.subspace). Install a checkout that ships it in the "
            "same environment, e.g. `pip install -e ../clifford_qc[openfermion]`."
        ) from exc
    return CliffordQCSubspaceAPI(
        run_acase=run_acase,
        determinant_excitations=determinant_excitations,
        occupied_spin_orbitals=occupied_spin_orbitals,
        dense_residual_norm=dense_residual_norm,
        adapt_warm_start=adapt_warm_start,
    )


def clifford_qc_available() -> bool:
    """Is the ADAPT-VQE surface (chemistry extra included) importable?"""
    try:
        load_clifford_qc()
    except ImportError:
        return False
    return True


def clifford_qc_subspace_available() -> bool:
    """Is the A-CASE surface importable?  It does not need the chemistry extra."""
    try:
        load_clifford_qc_core()
        load_clifford_qc_subspace()
    except ImportError:
        return False
    return True


def _to_clifford_fcidump(ham: MaterialHamiltonian) -> Any:
    api = load_clifford_qc_core()
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
    return api.FCIDump(
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


def _ansatz_state(api: CliffordQCAPI, model: Any, pool: list[Any], result: Any) -> Any:
    """Prepare the converged ADAPT ansatz state.

    Both the 1-RDM and the residual gradient are read off this one state. They
    used to prepare it separately from the identical (model, operators,
    parameters) triple, which ran the dominant cost of the solve twice.
    """
    by_label = {op.label: op for op in pool}
    try:
        chosen = [by_label[label] for label in result.labels]
    except KeyError as exc:
        raise RuntimeError(f"clifford_qc ADAPT result references unknown pool label {exc.args[0]!r}") from exc
    return api.ExactMVBackend().state(api.ansatz_program(model, chosen), result.parameters)


def _spatial_one_rdm(api: CliffordQCAPI, model: Any, rho: Any) -> list[list[float]]:
    n_spatial = model.n // 2
    gamma = np.zeros((n_spatial, n_spatial), dtype=float)
    for p in range(n_spatial):
        for q in range(n_spatial):
            value = 0.0
            for spin in (0, 1):
                op = api.cdag_op(model.n, 2 * p + spin) * api.c_op(model.n, 2 * q + spin)
                value += float(api.expectation(rho, op).real)
            gamma[p, q] = value
    # Numerical noise can make the exact real-sector result microscopically asymmetric.
    gamma = 0.5 * (gamma + gamma.T)
    return gamma.tolist()


def _residual_gradient(api: CliffordQCAPI, model: Any, pool: list[Any], result: Any, rho: Any) -> float:
    selected = set(result.labels)
    candidates = [i for i, op in enumerate(pool) if op.label not in selected]
    if not candidates:
        return 0.0
    bank = api.CommutatorBank(
        model.hamiltonian, [op.word for op in pool], [op.label for op in pool]
    )
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
        api = load_clifford_qc()
        data = _to_clifford_fcidump(hamiltonian)
        model = api.model_from_fcidump(data, name="qe-adapt-active-space")
        pool = api.excitation_pool(model.n, data.n_electrons)
        if not pool:
            raise ValueError("clifford_qc excitation pool is empty for this active-space sector")

        result = api.run_adapt(
            model,
            pool,
            max_operators=self.max_adapt_iterations,
            threshold=self.gradient_threshold,
            optimizer_method=self.optimizer_method,
            maxiter=self.maxiter,
            compute_exact_reference=self.compute_exact_reference,
            track_exact_scores=True,
        )
        rho = _ansatz_state(api, model, pool, result)
        one_rdm = _spatial_one_rdm(api, model, rho)
        natural = sorted((float(x) for x in np.linalg.eigvalsh(np.asarray(one_rdm))), reverse=True)
        residual = _residual_gradient(api, model, pool, result, rho)

        exact_ha = result.exact_ground_energy
        total_ev = float(result.energy) * HARTREE_TO_EV
        constant_ev = float(data.core_energy) * HARTREE_TO_EV
        hf_ha = api.ExactMVBackend().expectation(model.reference, model.hamiltonian, ())
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
            # The same number as metadata["residual_gradient"], in the
            # method-agnostic field the outer loop reads. Hartree, like every
            # quantity crossing the clifford_qc boundary.
            residual=residual,
            residual_kind="adapt_pool_gradient",
            metadata={
                "backend_project": "gutama/clifford_qc",
                "backend_api": "clifford_qc.algorithms.adapt.run_adapt",
                "scientific_status": "delegated_to_clifford_qc",
                "stopped_reason": result.stopped_reason,
                "residual_gradient": residual,
                "residual_unit": "Hartree",
                "exact_ground_energy_ev": (None if exact_ha is None else float(exact_ha) * HARTREE_TO_EV),
                "pool_size": len(pool),
                "energy_interchange_unit": "Hartree",
                "integral_convention": "chemist_(pq|rs)",
            },
        )


def _projected_one_rdm(api: CliffordQCCoreAPI, model: Any, solved: Any) -> list[list[float]]:
    """Spatial 1-RDM of a Ritz root, read through the projected-observable route.

    A-CASE never forms the Ritz state, so the density matrix is assembled from
    ``SubspaceResult.expectation`` rather than from a state vector. That method
    requires a Hermitian observable, and ``c†_p c_q`` is not Hermitian for
    ``p != q``, so the off-diagonal element is measured as the Hermitian
    combination ``(c†_p c_q + c†_q c_p) / 2`` -- which is exactly the symmetric
    part the ADAPT path also reports.
    """
    n_spatial = model.n // 2
    gamma = np.zeros((n_spatial, n_spatial), dtype=float)
    for p in range(n_spatial):
        for q in range(p, n_spatial):
            observable = None
            for spin in (0, 1):
                P, Q = 2 * p + spin, 2 * q + spin
                term = api.cdag_op(model.n, P) * api.c_op(model.n, Q)
                if P != Q:
                    term = 0.5 * (term + api.cdag_op(model.n, Q) * api.c_op(model.n, P))
                observable = term if observable is None else observable + term
            value = float(solved.expectation(observable))
            gamma[p, q] = value
            gamma[q, p] = value
    return gamma.tolist()


def _real_vector(values: Any, tolerance: float = 1e-9) -> list[float] | None:
    """Real part of a Ritz vector, or ``None`` when it is not really real.

    Silently discarding an imaginary part would turn a complex-valued result
    into a plausible-looking real one, so the caller records nothing instead.
    """
    array = np.asarray(values)
    if np.max(np.abs(array.imag)) > tolerance:
        return None
    return [float(x) for x in array.real]


class _CliffordQCSubspaceSolver(QuantumSolver):
    """Shared A-CASE machinery; subclasses only choose the reference state.

    The growth is ``clifford_qc.subspace.run_acase``: a Rayleigh-Ritz solve in
    ``span{A_i|psi>}`` whose basis is grown one generator at a time from the
    symmetry-preserving determinant excitations of the active space.  No
    variational parameters exist, so ``operator_parameters`` and
    ``operator_gradients`` stay empty rather than being filled with quantities
    that merely resemble them; the growth diagnostics go to ``metadata``.
    """

    solver_type = "clifford_qc_acase"
    reference_kind = "hartree_fock_determinant"

    def __init__(
        self,
        max_basis_size: int = 10,  # retained basis size, identity included
        max_excitation_rank: int = 2,
        leakage_tol: float | None = 1e-10,
        roots: int = 1,
        min_lowering: float | None = None,
        residual_threshold: float = 1e-6,
        compute_ritz_residual: bool = True,
    ) -> None:
        self.max_basis_size = int(max_basis_size)
        self.max_excitation_rank = int(max_excitation_rank)
        self.leakage_tol = None if leakage_tol is None else float(leakage_tol)
        self.roots = int(roots)
        self.min_lowering = None if min_lowering is None else float(min_lowering)
        self.residual_threshold = float(residual_threshold)
        self.compute_ritz_residual = bool(compute_ritz_residual)

    def _reference_state(self, api: CliffordQCCoreAPI, sub: CliffordQCSubspaceAPI,
                         model: Any, data: Any) -> tuple[Any, dict[str, Any]]:
        """The state A-CASE grows around, plus what to record about it."""
        return api.ExactMVBackend().state(model.reference, ()), {}

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
            raise ValueError("A-CASE requires a Hermitian restricted Hamiltonian")
        # A-CASE needs the model/state surface only: requiring the ADAPT
        # excitation pool here would make the chemistry extra a dependency of a
        # method that never touches it.
        api = load_clifford_qc_core()
        sub = load_clifford_qc_subspace()
        data = _to_clifford_fcidump(hamiltonian)
        model = api.model_from_fcidump(data, name="qe-adapt-active-space")

        rho, reference_metadata = self._reference_state(api, sub, model, data)
        occupied = sub.occupied_spin_orbitals(model)
        candidates = sub.determinant_excitations(
            model.n, occupied, max_rank=self.max_excitation_rank
        )
        if not candidates:
            raise ValueError(
                "no symmetry-preserving determinant excitations exist for this "
                "active-space sector, so there is no subspace to grow "
                f"(occupied spin orbitals: {occupied})"
            )

        growth_options: dict[str, Any] = {
            # ``run_acase``'s max_size counts *additions* to the initial
            # identity basis; max_basis_size is the retained basis size the
            # caller actually reads back, so the identity is subtracted here
            # rather than leaving the reported size one larger than requested.
            "max_size": min(max(1, self.max_basis_size - 1), len(candidates)),
            "roots": self.roots,
            "leakage_tol": self.leakage_tol,
        }
        if self.min_lowering is not None:
            growth_options["min_lowering"] = self.min_lowering
        result = sub.run_acase(rho, model.hamiltonian, candidates, **growth_options)

        solved = result.result
        # The growth objective aggregates tracked roots; the reported energy is
        # always the ground Ritz value, so a state-averaged run does not quietly
        # report an average as if it were a ground-state energy.
        total_ha = float(solved.ground_energy)
        residual = None
        if self.compute_ritz_residual:
            retained = [result.bank.generator(index) for index in result.indices]
            residual = float(sub.dense_residual_norm(rho, model.hamiltonian, retained, solved, 0))

        one_rdm = _projected_one_rdm(api, model, solved)
        natural = sorted((float(x) for x in np.linalg.eigvalsh(np.asarray(one_rdm))), reverse=True)
        hf_ha = api.ExactMVBackend().expectation(model.reference, model.hamiltonian, ())
        # ``expectation`` pairs two multivectors, so the Pauli-sum Hamiltonian is
        # converted rather than passed as-is.
        reference_ha = float(api.expectation(rho, model.hamiltonian.to_mv()).real)

        constant_ev = float(data.core_energy) * HARTREE_TO_EV
        total_ev = total_ha * HARTREE_TO_EV

        # Growth that never left the reference is the honest reading of a
        # subspace that could not improve on its starting state -- it is
        # reported, not smoothed over, because for a correlated reference it
        # means the excitation family is stationary there rather than exact.
        basis_size = len(result.labels)
        if residual is None:
            # Fail closed: only an exhausted candidate pool (a complete basis in
            # this family) certifies convergence without a residual.
            converged = result.stopped_reason == "candidate pool exhausted"
        else:
            converged = residual < self.residual_threshold

        metadata: dict[str, Any] = {
            "backend_project": "gutama/clifford_qc",
            "backend_api": "clifford_qc.subspace.run_acase",
            "scientific_status": "delegated_to_clifford_qc",
            "method_family": "projected_subspace_eigensolver",
            "reference": self.reference_kind,
            "reference_energy_ev": round(reference_ha * HARTREE_TO_EV, 10),
            "subspace_lowering_ev": round((reference_ha - total_ha) * HARTREE_TO_EV, 10),
            "grew_beyond_reference": basis_size > 1,
            "stopped_reason": result.stopped_reason,
            "basis_size": basis_size,
            "candidate_pool_size": len(candidates),
            "max_excitation_rank": self.max_excitation_rank,
            "condition_number": float(solved.condition_number),
            "effective_rank": int(solved.effective_rank),
            "roots": self.roots,
            "root_energies_ev": [float(x) * HARTREE_TO_EV for x in solved.energies],
            "predicted_lowerings": [float(rec.predicted_lowering) for rec in result.records],
            "ritz_vector": _real_vector(solved.ritz_vector(0)),
            "ritz_residual_norm": residual,
            "residual_unit": "Hartree",
            "energy_interchange_unit": "Hartree",
            "integral_convention": "chemist_(pq|rs)",
            "growth_seconds": result.resources.get("growth_seconds"),
            "word_universe": result.resources.get("word_universe"),
        }
        metadata.update(reference_metadata)

        return QuantumRunResult(
            energy_ev=round(total_ev, 10),
            electronic_energy_ev=round(total_ev - constant_ev, 10),
            constant_energy_ev=round(constant_ev, 10),
            correlation_energy_ev=round((total_ha - float(hf_ha)) * HARTREE_TO_EV, 10),
            solver_type=self.solver_type,
            n_orbitals=hamiltonian.n_orbitals,
            n_electrons=float(data.n_electrons),
            n_spin_orbitals=model.n,
            converged=converged,
            selected_operators=list(result.labels),
            operator_gradients=[],  # a Ritz growth has no operator gradients
            operator_parameters=[],  # nor variational parameters
            iteration_energies=[float(x) * HARTREE_TO_EV for x in result.energy_history],
            one_rdm=one_rdm,
            natural_occupations=[round(x, 10) for x in natural],
            residual=residual,
            residual_kind=("ritz_residual_norm" if residual is not None else ""),
            metadata=metadata,
        )


class CliffordQCACASESolver(_CliffordQCSubspaceSolver):
    """A-CASE grown around the active space's Hartree-Fock determinant.

    The candidate generators are the particle-number and S_z conserving
    determinant excitations of that reference, so the subspace never leaves the
    sector the active space declared.
    """


class CliffordQCADAPTACASESolver(_CliffordQCSubspaceSolver):
    """ADAPT-VQE first, then A-CASE grown around the optimized ADAPT state.

    This is ``clifford_qc.subspace.adapt_warm_start``: the composition the
    sibling project supports directly, rather than two runs glued together
    here.

    A warmer reference is not automatically a better answer, and this solver
    does not pretend otherwise.  A converged ADAPT state is stationary against
    the very excitation family A-CASE would grow with, so the subspace can
    refuse to grow at all (``grew_beyond_reference=False``) while still sitting
    above the exact energy -- the Ritz residual, not the absence of growth, is
    what says whether that point is converged.
    """

    solver_type = "clifford_qc_adapt_acase"
    reference_kind = "adapt_vqe_state"

    def __init__(
        self,
        adapt_max_operators: int = 4,
        adapt_gradient_threshold: float = 1e-6,
        optimizer_method: str = "auto",
        maxiter: int = 350,
        **subspace_options: Any,
    ) -> None:
        super().__init__(**subspace_options)
        self.adapt_max_operators = int(adapt_max_operators)
        self.adapt_gradient_threshold = float(adapt_gradient_threshold)
        self.optimizer_method = optimizer_method
        self.maxiter = int(maxiter)

    def _reference_state(self, api: CliffordQCCoreAPI, sub: CliffordQCSubspaceAPI,
                         model: Any, data: Any) -> tuple[Any, dict[str, Any]]:
        # The warm start runs ADAPT first, so this path does need the pool.
        pool = load_clifford_qc().excitation_pool(model.n, data.n_electrons)
        if not pool:
            raise ValueError("clifford_qc excitation pool is empty for this active-space sector")
        rho, adapt = sub.adapt_warm_start(
            model,
            pool,
            max_operators=self.adapt_max_operators,
            threshold=self.adapt_gradient_threshold,
            optimizer_method=self.optimizer_method,
            maxiter=self.maxiter,
        )
        return rho, {
            "warm_start_api": "clifford_qc.subspace.adapt_warm_start",
            "adapt_operators": list(adapt.labels),
            "adapt_parameters": [float(x) for x in adapt.parameters],
            "adapt_energy_ev": round(float(adapt.energy) * HARTREE_TO_EV, 10),
            "adapt_stopped_reason": adapt.stopped_reason,
            "adapt_pool_size": len(pool),
        }
