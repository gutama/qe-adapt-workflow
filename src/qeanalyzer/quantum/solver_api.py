"""Stable quantum-solver facade for QE-ADAPT workflow integration.

This module is the single place that maps a solver name onto a solver class.
The scientific implementations are owned by the sibling ``clifford_qc``
project, so this module deliberately contains no second copy of ADAPT-VQE or
A-CASE -- and no second copy of the factory either.

Adding a method is one :class:`SolverSpec` in ``_SPECS``.  A spec carries the
aliases, what the method actually is, what it needs installed, and its
scientific status, so ``describe_solvers()`` can answer "what can this workflow
drive?" without a second list that drifts out of date.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from qeanalyzer.quantum.adapt_bridge import (
    ExactDiagonalizationSolver,
    QuantumRunResult,
    QuantumSolver,
    SimulatedADAPTVQESolver,
)
from qeanalyzer.quantum.clifford_bridge import (
    CliffordQCACASESolver,
    CliffordQCADAPTACASESolver,
    CliffordQCADAPTSolver,
)
from qeanalyzer.quantum.hamiltonian import MaterialHamiltonian

# Backwards-compatible public name: real ADAPT, delegated to clifford_qc. It is
# the class itself, so isinstance(solver, ADAPTVQESolver) behaves.
ADAPTVQESolver = CliffordQCADAPTSolver


@dataclass(frozen=True)
class SolverSpec:
    """One method this workflow can drive, and what it is honest about."""

    name: str
    factory: Callable[..., QuantumSolver]
    summary: str
    scientific_status: str
    aliases: frozenset[str] = field(default_factory=frozenset)
    requires: str = ""
    accepts_options: bool = True

    @property
    def all_names(self) -> frozenset[str]:
        return self.aliases | {self.name}


def _comparison_solver(**options: Any) -> QuantumSolver:
    """Build the multi-arm composite.

    Imported inside the factory because the composite builds its arms through
    ``create_quantum_solver``; keeping the module-level dependency one-way means
    the registry stays importable on its own.
    """
    from qeanalyzer.quantum.comparison import ComparisonSolver

    return ComparisonSolver(**options)


_SPECS: tuple[SolverSpec, ...] = (
    SolverSpec(
        name="exact",
        factory=ExactDiagonalizationSolver,
        summary="Small-space exact FCI reference by explicit second quantization.",
        scientific_status="exact_reference",
        aliases=frozenset({"fci", "exact_diagonalization", "ed"}),
        accepts_options=False,
    ),
    SolverSpec(
        name="adapt_vqe",
        factory=CliffordQCADAPTSolver,
        summary="ADAPT-VQE: adaptively grown unitary ansatz, variationally optimized.",
        scientific_status="delegated_to_clifford_qc",
        aliases=frozenset({"adapt", "clifford_adapt", "clifford_qc_adapt"}),
        requires="clifford_qc",
    ),
    SolverSpec(
        name="acase",
        factory=CliffordQCACASESolver,
        summary=(
            "A-CASE: Rayleigh-Ritz in a subspace grown around the reference "
            "determinant; no variational parameters."
        ),
        scientific_status="delegated_to_clifford_qc",
        aliases=frozenset({"a_case", "clifford_acase", "subspace"}),
        requires="clifford_qc.subspace",
    ),
    SolverSpec(
        name="adapt_acase",
        factory=CliffordQCADAPTACASESolver,
        summary="ADAPT-VQE state used as the A-CASE reference (warm start).",
        scientific_status="delegated_to_clifford_qc",
        aliases=frozenset({"adapt_warm_start", "warm_start_acase", "acase_adapt"}),
        requires="clifford_qc.subspace",
    ),
    SolverSpec(
        name="compare",
        factory=_comparison_solver,
        summary="Run several methods on one Hamiltonian and record every arm.",
        scientific_status="comparison_record",
        aliases=frozenset({"multi", "multi_arm", "comparison"}),
    ),
    SolverSpec(
        name="simulated_adapt",
        factory=SimulatedADAPTVQESolver,
        summary="Monotone fake trajectory for workflow plumbing tests only.",
        scientific_status="workflow_mock",
        aliases=frozenset({"mock_adapt", "workflow_mock"}),
    ),
)

_BY_NAME: dict[str, SolverSpec] = {}
for _spec in _SPECS:
    for _alias in _spec.all_names:
        if _alias in _BY_NAME:
            raise RuntimeError(f"duplicate solver alias {_alias!r} in the registry")
        _BY_NAME[_alias] = _spec

_BUILTIN_SPECS = _SPECS

# Names that used to mean one method and now cannot.  Answering them with a
# guess is how a run ends up labelled as the method it was not.
_AMBIGUOUS: dict[str, str] = {
    "vqe": (
        "'vqe' is ambiguous now that more than one variational method is "
        "available: 'adapt_vqe' is ADAPT-VQE, 'acase' is the A-CASE subspace "
        "eigensolver, and clifford_qc also implements a fixed-ansatz VQE that "
        "this workflow does not bridge. Name the method you mean."
    ),
}

# Retained for callers that imported the alias sets directly; derived from the
# registry so they cannot drift away from what the factory actually accepts.
EXACT_ALIASES = _BY_NAME["exact"].all_names
ADAPT_ALIASES = _BY_NAME["adapt_vqe"].all_names
ACASE_ALIASES = _BY_NAME["acase"].all_names
MOCK_ALIASES = _BY_NAME["simulated_adapt"].all_names


def normalize_solver_type(solver_type: str) -> str:
    """Canonical registry name for an alias, so ``adapt-vqe`` and ``ADAPT_VQE`` agree."""
    key = solver_type.lower().replace("-", "_").strip()
    if key in _AMBIGUOUS:
        raise ValueError(_AMBIGUOUS[key])
    spec = _BY_NAME.get(key)
    if spec is None:
        raise ValueError(
            f"Unknown quantum solver type {solver_type!r}. "
            f"Available: {', '.join(solver_names())}."
        )
    return spec.name


def solver_names() -> list[str]:
    """Canonical names, in registry order."""
    return [spec.name for spec in _SPECS]


def available_solvers() -> tuple[SolverSpec, ...]:
    return _SPECS


def describe_solvers() -> str:
    """Human-readable table of methods, aliases and scientific status."""
    lines = [f"{'name':<16}{'status':<28}{'requires':<22}summary"]
    for spec in _SPECS:
        lines.append(
            f"{spec.name:<16}{spec.scientific_status:<28}"
            f"{(spec.requires or '-'):<22}{spec.summary}"
        )
        others = sorted(spec.aliases)
        if others:
            lines.append(f"{'':<16}aliases: {', '.join(others)}")
    return "\n".join(lines)


def create_quantum_solver(solver_type: str = "exact", **solver_options: Any) -> QuantumSolver:
    """Build a named solver without duplicating any of them in this repository.

    ``adapt_vqe`` and ``acase`` select the real :mod:`clifford_qc` backends,
    ``adapt_acase`` composes them, ``compare`` runs several arms at once, and
    ``simulated_adapt`` is an explicit workflow-only mock that must never be
    used as scientific evidence.  Names are normalized, so ``adapt-vqe`` and
    ``ADAPT_VQE`` are the same solver.

    ``solver_options`` are the chosen solver's constructor arguments.  The exact
    solver has none, so passing any is an error rather than a silent no-op.
    """
    spec = _BY_NAME[normalize_solver_type(solver_type)]
    if solver_options and not spec.accepts_options:
        raise TypeError(
            f"solver_type={solver_type!r} takes no options, but got "
            f"{sorted(solver_options)}. Exact diagonalization has nothing to tune; "
            "pass solver_type='adapt_vqe' if these are ADAPT settings."
        )
    return spec.factory(**solver_options)


def solve_active_space(
    hamiltonian: MaterialHamiltonian,
    active_space: Any = None,
    solver_type: str = "exact",
    **solver_options: Any,
) -> QuantumRunResult:
    """Build the named solver and run it once.

    ``solver_options`` configure the solver itself and are forwarded only to its
    constructor.  Forwarding them to ``solve`` as well meant an option the
    constructor ignored was swallowed by ``solve(**kwargs)``, so a caller's
    setting could take effect in neither place without raising.
    """
    solver = create_quantum_solver(solver_type, **solver_options)
    return solver.solve(hamiltonian, active_space=active_space)


def register_solver(spec: SolverSpec) -> None:
    """Add a method to the registry at runtime (tests, downstream projects).

    Registration is append-only and refuses to shadow an existing name: a silent
    override would make ``create_quantum_solver('adapt_vqe')`` mean something
    different depending on what had been imported.
    """
    global _SPECS
    for alias in spec.all_names:
        if alias in _BY_NAME:
            raise ValueError(f"solver alias {alias!r} is already registered")
        if alias in _AMBIGUOUS:
            raise ValueError(f"solver alias {alias!r} is reserved as ambiguous")
    _SPECS = _SPECS + (spec,)
    for alias in spec.all_names:
        _BY_NAME[alias] = spec


def unregister_solver(name: str) -> None:
    """Remove a runtime-registered method; built-in specs are not removable."""
    global _SPECS
    spec = _BY_NAME.get(name)
    if spec is None:
        raise KeyError(name)
    if spec in _BUILTIN_SPECS:
        raise ValueError(f"built-in solver {spec.name!r} cannot be unregistered")
    _SPECS = tuple(s for s in _SPECS if s is not spec)
    for alias in [a for a, s in _BY_NAME.items() if s is spec]:
        del _BY_NAME[alias]


__all__ = [
    "ACASE_ALIASES",
    "ADAPTVQESolver",
    "ADAPT_ALIASES",
    "CliffordQCACASESolver",
    "CliffordQCADAPTACASESolver",
    "CliffordQCADAPTSolver",
    "EXACT_ALIASES",
    "ExactDiagonalizationSolver",
    "MOCK_ALIASES",
    "QuantumRunResult",
    "QuantumSolver",
    "SimulatedADAPTVQESolver",
    "SolverSpec",
    "available_solvers",
    "create_quantum_solver",
    "describe_solvers",
    "normalize_solver_type",
    "register_solver",
    "solve_active_space",
    "solver_names",
    "unregister_solver",
]
