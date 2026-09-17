# Scientific Boundaries and Ownership

This document freezes the scientific/API boundaries between `qe-adapt-workflow` and the sibling [`gutama/clifford_qc`](https://github.com/gutama/clifford_qc) project.

## 1. Repository ownership

`qe-adapt-workflow` owns:

- Quantum ESPRESSO input/output/XML parsing;
- coherent run-source resolution;
- diagnostics, plotting, provenance and workflow ledgers;
- deterministic QE next-run/recovery policies;
- active-space *selection metadata* from QE results;
- finite restricted Hamiltonian interchange;
- FCIDUMP serialization;
- outer-loop orchestration.

`clifford_qc` owns:

- Pauli/Clifford operator algebra;
- Jordan-Wigner quantum-model execution;
- ADAPT-VQE selection and variational optimization;
- A-CASE subspace growth, projected matrix elements and Ritz solves;
- finite-shot selection/measurement machinery;
- other quantum algorithms and subspace methods.

The QE project must not maintain a second scientific implementation of any of them. Adding a method here means adding a *bridge* -- a `QuantumSolver` that translates this repository's Hamiltonian into the backend's model and the backend's result into a `QuantumRunResult` -- never a second engine.

## 2. Three different Hamiltonian paths

They must not be conflated.

### A. Parameterized band-derived model

`build_band_model_hamiltonian()` averages selected QE Kohn-Sham eigenvalues using k-point weights and optionally adds caller-supplied U/V parameters.

This is useful for:

- software integration tests;
- toy models;
- controller prototypes;
- quick effective-model experiments where the approximation is explicitly intended.

It is **not** an ab-initio QE-to-FCIDUMP conversion.

### B. Explicit finite integral Hamiltonian

`build_integral_hamiltonian(h1, h2, ...)` is the scientific interchange boundary when a consistent finite basis and interaction tensor already exist.

Internal convention:

```text
h2[p][q][r][s] = (pq|rs)   # chemist notation
```

and

```text
H2 = 1/2 sum_pqrs sum_sigma,tau (pq|rs)
     a†_(p,sigma) a†_(r,tau) a_(s,tau) a_(q,sigma)
```

### C. Physical QE materials downfolding

Target architecture:

```text
QE
 → Wannier90 / another declared finite localized basis
 → consistent one-body model
 → screened/bare interaction construction (e.g. cRPA or explicit ERIs)
 → explicit double-counting convention
 → build_integral_hamiltonian
 → FCIDUMP / clifford_qc
```

This route is not yet implemented end to end.

## 3. Why Kohn-Sham bands are not FCIDUMP orbitals

A periodic band is a family `psi_{n,k}`, not a single finite localized spatial orbital. A selected band index therefore cannot silently become one FCIDUMP orbital.

Band-based selectors record their representation as periodic band indices. A finite many-body Hamiltonian requires a declared mapping such as Wannier localization, finite k-space truncation, embedding orbitals, or another explicit basis definition.

## 4. FCIDUMP units

FCIDUMP has no unit field. `qeanalyzer` therefore uses the conventional boundary:

```text
FCIDUMP numerical energies = Hartree
```

Writers convert from the internal `MaterialHamiltonian.energy_unit` to Hartree. Readers return `energy_unit="Hartree"`.

## 5. Particle-number sectors

DFT smearing can give a fractional active electron count. A fixed-sector FCI/ADAPT calculation cannot infer the intended integer sector from that number.

The code therefore accepts only values numerically equal to an integer and otherwise raises. The user/downfolding policy must make the particle-number choice explicitly.

## 6. Spin restrictions

The restricted finite-Hamiltonian path assumes one spatial orbital with alpha/beta partners. It does not currently cover:

- LSDA spin-dependent band indexing;
- noncollinear spinors;
- spin-orbit coupled spinors;
- unrestricted FCIDUMP (`IUHF=1`);
- complex integrals.

Those cases fail explicitly instead of being silently coerced.

## 7. Real ADAPT-VQE

The public `ADAPTVQESolver` is an alias of `CliffordQCADAPTSolver`, which calls:

```python
clifford_qc.algorithms.adapt.run_adapt
```

with the `clifford_qc.models.chemistry.excitation_pool` pool and a model constructed from the same restricted FCIDUMP convention.

The adapter additionally evaluates the final spatial 1-RDM and the final residual commutator-gradient maximum from the actual optimized ADAPT state.

The old synthetic trajectory is exposed only as `SimulatedADAPTVQESolver` and is marked non-scientific.

## 7a. A-CASE, and why it is a different method

`CliffordQCACASESolver` calls:

```python
clifford_qc.subspace.run_acase
```

on the same restricted FCIDUMP convention, with `clifford_qc.subspace.determinant_excitations` as the candidate generators. A-CASE is **not** a variant of ADAPT-VQE:

| | ADAPT-VQE | A-CASE |
|---|---|---|
| object grown | unitary ansatz `exp(theta_k A_k)` | linear subspace `span{A_i\|psi>}` |
| how the answer is found | parameter optimization | Rayleigh-Ritz generalized eigenproblem |
| convergence measure | pool commutator gradient | Ritz residual norm |
| reported here as | `residual_kind="adapt_pool_gradient"` | `residual_kind="ritz_residual_norm"` |

Consequences the bridge is explicit about:

- **No variational parameters exist**, so `operator_parameters` and `operator_gradients` stay empty rather than being filled with predicted lowerings that merely resemble gradients. Growth diagnostics live in `metadata`.
- **The 1-RDM is read through the projected-observable route** (`SubspaceResult.expectation`), never by forming the Ritz state. `expectation` requires a Hermitian observable, so an off-diagonal element is measured as `(c†_p c_q + c†_q c_p)/2` -- exactly the symmetric part the ADAPT path also reports.
- **The Ritz residual is a dense validation-tier quantity** (`clifford_qc.subspace.dense_residual_norm`): it reconstructs the state explicitly, so its cost is exponential in the active space, like this repository's own exact FCI reference. `compute_ritz_residual=False` turns it off, and the solver then reports no residual at all rather than a cheaper number under the same name.
- **Convergence is fail-closed.** With a residual, converged means residual below threshold. Without one, only an exhausted candidate pool -- a complete basis in that excitation family -- counts as converged; growth that merely stopped improving does not.
- **The chemistry extra is not a dependency of this path.** `openfermion` is needed by ADAPT's excitation pool; A-CASE builds determinant excitations natively, so the bridge loads only `CliffordQCCoreAPI` (model, state, fermion operators) and leaves the pool to the methods that use it.

## 7b. ADAPT-VQE warm-started A-CASE

`CliffordQCADAPTACASESolver` is `clifford_qc.subspace.adapt_warm_start`: ADAPT-VQE runs first, and its optimized state becomes the A-CASE reference.

A warmer reference is **not** automatically a better answer, and the composition must not be presented as an improvement by construction:

- A converged ADAPT state is stationary against the same excitation family A-CASE grows with. For an anti-Hermitian generator `A`, the Ritz coupling `<psi|H A|psi>` is half the ADAPT gradient `<psi|[H,A]|psi>`, so directions ADAPT has already flattened contribute no first-order lowering. The subspace can then decline to grow at all -- `grew_beyond_reference=False` -- while still sitting above the exact energy.
- The upstream project has replicated a case where a *better* ADAPT reference produced a *worse* A-CASE subspace (`benchmarks/run_warm_start_replication.py`, Kendall tau of -1 across an operator ladder).

The bridge therefore reports the reference energy, the lowering the subspace actually achieved, and whether it grew at all -- and reads convergence from the residual, never from the absence of growth.

## 7c. Multi-arm comparison records

`compare_solvers` / `ComparisonSolver` run several methods on one Hamiltonian and keep every arm. The record is evidence, not a verdict:

- Every arm is fingerprinted against one `hamiltonian_sha256`, and an arm that answered a different orbital count or particle-number sector is refused rather than tabulated next to the others.
- Ranking is the variational reading -- lowest energy on the same Hamiltonian and sector -- and is restricted to arms that are scientific results. `SimulatedADAPTVQESolver` derives its numbers from the exact solution and would frequently "win", so a `workflow_mock` arm is recorded, marked `rankable=false`, and never selected.
- **Budget matching is never inferred.** `matched_budget_declared` is true only when the caller passed a `budget_note`; otherwise the summary says the arms were not matched on cost. Wall time is recorded as provenance and is not a hardware cost model or a quantum resource count.
- A comparison is labelled `scientific_status="comparison_record"`, and the whole record is carried into the outer-loop ledger, so an iteration driven by one arm still shows what every other method said.

## 8. Feedback policies

The existing feedback rules are controller heuristics. They do not constitute a validated DFT+many-body self-consistency functional.

Every built-in decision carries:

```text
scientific_status = experimental_heuristic
validated_physical_self_consistency = false
```

In particular:

- `OccupationFeedbackPolicy` changes smearing/mixing; it does not inject a correlated RDM into the Kohn-Sham equations.
- `ActiveSpaceFeedbackPolicy` changes requested band count heuristically.
- `HubbardUFeedbackPolicy` is a legacy linear toy response and is not a substitute for cRPA/linear-response U or a validated orbital-to-species map.

## 9. Outer-loop convergence

Missing information is not success.

If `require_rdm=True`, missing or shape-incompatible 1-RDMs fail the RDM criterion. If `require_gradient=True`, an unavailable residual ADAPT gradient fails the gradient criterion. Absence is never silently converted to zero.

The third criterion is the **solver-reported residual**, not an ADAPT gradient. `ConvergenceCriteria.gradient_tolerance` keeps its name so existing ledgers stay readable, but what it compares is `QuantumRunResult.residual`, whose meaning per iteration is recorded as `quantum_residual_kind`. Residuals cross the `clifford_qc` boundary in **Hartree**, so that tolerance is in Hartree while `energy_tolerance_ev` is in eV. A solver that cannot supply a residual reports `None`, which is not a zero: it fails the criterion until the criterion is switched off.

`require_rdm=False` / `require_gradient=False` switch the criterion off entirely: the quantity is still measured and recorded in the ledger for provenance, but it cannot gate convergence. The flag is a statement about which criteria the workflow is closing on, not merely permission for the solver to omit one -- otherwise a solver that *did* report the quantity would still gate a loop the user had already excluded it from, and that loop could never terminate.

## 10. Source coherence

One `QERunResult` must describe one QE run. CLI discovery therefore scans one run directory plus the `outdir` locations QE writes XML to, and rejects ambiguous multiple inputs/outputs/XML files. It does not recursively combine files from several serial calculations.

Concretely, discovery resolves `<prefix>.save/` both directly in the run directory and one level below it, because `outdir` is normally a subdirectory of the run (`outdir='./tmp'`). The scan stops at that depth: anything deeper is a workflow parent, not one run.

Coherence is judged per run, not per file. QE >= 6.4 writes a single run's XML record twice -- `<outdir>/<prefix>.xml` and `<outdir>/<prefix>.save/data-file-schema.xml` -- so both map to the same `(outdir, prefix)` run identity and the canonical schema record is used. Two *different* prefixes in one directory remain ambiguous and are still rejected.

## 11. Periodic band sampling for energy windows

`EnergyWindowSelector` judges a band against the window at one of four samplings, recorded in the active-space metadata as `kpoint_mode_resolved`:

- `gamma` reads the band at the Gamma point, located by coordinate rather than assumed to be listed first.
- `average` uses the k-point-weighted mean over the Brillouin zone.
- `any` includes a band that enters the window anywhere in the zone.
- `auto` (the default) uses Gamma when the result contains it and the weighted average otherwise.

`auto` exists because Gamma is not always sampled: a shifted Monkhorst-Pack mesh has no k-point at (0, 0, 0), and a result built from `pw.out` alone carries no k-point coordinates at all. Requesting `gamma` explicitly still raises in both cases, since the caller asked for one specific k-point; `auto` records which sampling it actually used rather than leaving the choice implicit.

## 12. FCIDUMP symmetry provenance

`ORBSYM`/`ISYM` are provenance, not defaults. `write_fcidump` takes them from the caller, else from `MaterialHamiltonian.metadata` where `parse_fcidump` records them, and only falls back to all-ones/1 for a Hamiltonian that never carried point-group symmetry. Writing all-ones unconditionally would make a read-modify-write round-trip silently drop the symmetry blocking that consumers such as NECI and Dice use, changing the calculation the file describes.
