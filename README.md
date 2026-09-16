# qe-adapt-workflow

A reproducible analysis and workflow-control layer for serial **Quantum ESPRESSO** calculations, with an explicit bridge to correlated quantum solvers.

The repository is intentionally split from [`gutama/clifford_qc`](https://github.com/gutama/clifford_qc):

- **qe-adapt-workflow owns** QE parsing, diagnostics, run provenance, next-input generation, active-space *selection metadata*, finite-Hamiltonian interchange, solver selection/comparison, and outer-loop orchestration.
- **clifford_qc owns** ADAPT-VQE, A-CASE, Pauli/Clifford operator algebra, measurement/selection machinery, and related quantum algorithms.

This avoids maintaining a second scientific implementation of any of them here. The workflow is **not ADAPT-only**: it drives a registry of correlated methods that all speak one `QuantumSolver` interface, and it can run several of them on one Hamiltonian and keep every answer.

## Scientific status

The software framework is implemented, but not every path has the same physical status:

| Capability | Status |
|---|---|
| QE input/text/XML parsing | Implemented |
| QE run reports and convergence plots | Implemented |
| Deterministic relax → SCF → NSCF/recovery policies | Implemented prototype |
| Workflow ledger / local / Slurm execution | Implemented prototype |
| Small-space exact FCI reference solver | Implemented |
| FCIDUMP I/O | Implemented; **FCIDUMP boundary is Hartree** |
| Real ADAPT-VQE | Delegated to `clifford_qc.algorithms.adapt.run_adapt` |
| Real A-CASE subspace eigensolver | Delegated to `clifford_qc.subspace.run_acase` |
| ADAPT-warm-started A-CASE | Delegated to `clifford_qc.subspace.adapt_warm_start`; **a warmer reference is not automatically a better answer** |
| Multi-arm method comparison | Implemented; a **record**, not a benchmark verdict -- budget matching is never inferred |
| QE band-derived toy/effective model | Implemented, explicitly heuristic |
| Ab-initio QE → correlated Hamiltonian | **Not yet implemented**; requires Wannier/downfolding/integrals |
| Quantum → DFT feedback policies | Experimental controller heuristics, not a validated self-consistency functional |

See [`docs/SCIENTIFIC_BOUNDARIES.md`](docs/SCIENTIFIC_BOUNDARIES.md).

## Architecture

```text
Quantum ESPRESSO
  pw.in / pw.out / prefix.save/data-file-schema.xml
        │
        ▼
qeanalyzer
  parse → validate → QERunResult → report/plots
        │
        ├── deterministic next-QE policy ────────────┐
        │                                            │
        └── active-space selection metadata          │
                     │                               │
                     ▼                               │
        Hamiltonian construction boundary            │
          ├─ parameterized band model (heuristic)     │
          └─ explicit h1/(pq|rs) integrals (physical │
             interchange; Wannier/cRPA upstream)      │
                     │                               │
                     ▼                               │
                 FCIDUMP (Hartree)                   │
                     │                               │
                     ▼                               │
          solver registry (one Hamiltonian)          │
          ├─ exact FCI reference (in repo)           │
          ├─ ADAPT-VQE ──────┐                       │
          ├─ A-CASE ─────────┤ gutama/clifford_qc    │
          ├─ ADAPT → A-CASE ─┘ (warm start)          │
          └─ compare: several arms, all recorded     │
                     │                               │
                     ▼                               │
        energy / 1-RDM / residual (+ every arm)      │
                     │                               │
                     ▼                               │
          experimental feedback policy ──────────────┘
```

The original architecture plan remains in [`docs/ARCHITECTURE_IMPLEMENTATION_PLAN.md`](docs/ARCHITECTURE_IMPLEMENTATION_PLAN.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For plotting:

```bash
pip install -e '.[plot]'
```

For **real ADAPT-VQE and A-CASE**, install the sibling `clifford_qc` project in the same environment. The chemistry/openfermion extra supplies the excitation-pool adapter used by the ADAPT path:

```bash
pip install -e '../clifford_qc[openfermion]'
```

The `openfermion` extra is needed only by ADAPT's chemistry excitation pool: A-CASE builds its own determinant excitations and runs without it. If `clifford_qc` is not installed, solving with those names fails loudly and says what to install; the names themselves still resolve, so a script can be written before the backend is present. A separate `SimulatedADAPTVQESolver` exists only for workflow plumbing tests and is explicitly marked non-scientific.

## Correlated methods

ADAPT-VQE is one arm of the registry, not the whole workflow:

| name | what it is | needs |
|---|---|---|
| `exact` | small-space exact FCI, in this repository | -- |
| `adapt_vqe` | adaptively grown **unitary ansatz**, variationally optimized | `clifford_qc` |
| `acase` | **A-CASE**: Rayleigh-Ritz in `span{A_i\|psi>}` grown around the reference determinant | `clifford_qc.subspace` |
| `adapt_acase` | ADAPT-VQE state used as the A-CASE reference (warm start) | `clifford_qc.subspace` |
| `compare` | several arms on one Hamiltonian, every answer recorded | per arm |
| `simulated_adapt` | explicit non-scientific mock for plumbing tests | -- |

```bash
qeanalyzer solvers        # the same table, plus which backends are installed here
```

A-CASE is a different algorithm from ADAPT-VQE, not a second name for it: it grows a *linear subspace* and solves a projected generalized eigenproblem instead of optimizing ansatz parameters. It therefore reports a Ritz residual rather than a pool gradient, and no variational parameters at all. See [`docs/SCIENTIFIC_BOUNDARIES.md`](docs/SCIENTIFIC_BOUNDARIES.md) §7a.

### Combining methods

Three compositions are supported, and each is explicit about what it does *not* claim:

```python
from qeanalyzer.quantum import compare_solvers, create_quantum_solver

# 1. Sequential: ADAPT-VQE first, its optimized state as the A-CASE reference.
warm = create_quantum_solver("adapt_acase", adapt_max_operators=4, max_basis_size=12)
result = warm.solve(ham)
result.metadata["grew_beyond_reference"]   # did the subspace improve on ADAPT at all?

# 2. Side by side: one Hamiltonian, several methods, every arm kept.
comparison = compare_solvers(
    ham,
    [
        "exact",
        {"solver_type": "acase", "label": "acase_m12", "options": {"max_basis_size": 12}},
        {"solver_type": "adapt_vqe", "label": "adapt_8", "options": {"max_adapt_iterations": 8}},
    ],
    budget_note="basis 12 vs 8 ADAPT operators -- NOT cost-matched",
)
print(comparison.summary())

# 3. Inside the outer loop: drive on one arm, record them all in the ledger.
multi = create_quantum_solver("compare", arms=["exact", "acase"], selection="acase")
```

A warm start is **not** an improvement by construction. A converged ADAPT state is stationary against the same excitation family A-CASE grows with, so the subspace can decline to grow while still sitting above the exact energy; the bridge reports that rather than reading "no growth" as "converged". The upstream project has replicated a case where a better ADAPT reference gave a *worse* subspace.

A comparison is a record, not a verdict. Arms are ranked only as variational upper bounds on the same Hamiltonian and sector, the non-scientific mock is recorded but never ranked, and `matched_budget_declared` is true only when the caller declares the matching.

### Adding another method

One `SolverSpec`, registered once:

```python
from qeanalyzer.quantum import SolverSpec, register_solver

register_solver(SolverSpec(
    name="my_method",
    factory=MySolver,                 # implements QuantumSolver.solve -> QuantumRunResult
    summary="what it actually is",
    scientific_status="delegated_to_somewhere",
    requires="my_backend",
))
```

Registration is append-only and refuses to shadow an existing name, so `create_quantum_solver("adapt_vqe")` cannot come to mean something else depending on import order.

## CLI

### Analyze a single coherent QE run

```bash
qeanalyzer report run_dir/
qeanalyzer dump run_dir/ -o result.json
qeanalyzer plot run_dir/ -o convergence.png
```

Run discovery is deliberately conservative. `qeanalyzer` does **not** recursively choose the first `.in`, `.out`, and XML file under a multi-run workflow directory. Ambiguous bundles fail and require one run directory or explicit matching files.

### Generate the next QE input

```bash
qeanalyzer next 001_scf/ \
  --policy scf_to_nscf \
  -o 002_nscf/ \
  --ledger workflow.json
```

### Inspect workflow provenance

```bash
qeanalyzer history workflow.json
qeanalyzer validate workflow.json
qeanalyzer plot-history workflow.json -o history.png
```

### FCIDUMP

`FCIDUMP` has no unit field. This project therefore follows standard quantum-chemistry convention and **always writes FCIDUMP numerical energies in Hartree**. Imported FCIDUMP data are also labelled Hartree.

The current CLI command:

```bash
qeanalyzer export-fcidump run_dir/ \
  --active-method band_index --band-start 1 --band-end 4 \
  -u 2.5 -v 0.5 -o model.FCIDUMP
```

constructs a **parameterized QE band-derived effective model**: selected Kohn-Sham band energies are k-point-weight averaged and placed on a diagonal one-body model, while `U`/`V` are user-supplied parameters. It is useful for workflow tests and toy models but **is not an ab-initio QE→FCIDUMP conversion**.

A physical materials workflow should instead be:

```text
QE
 → Wannier90 / finite localized basis
 → interaction construction (e.g. cRPA or explicit ERIs)
 → build_integral_hamiltonian(h1, h2, ...)
 → FCIDUMP / clifford_qc
```

The code deliberately does not treat a Kohn-Sham Hamiltonian plus arbitrary bare interactions as an ab-initio many-body Hamiltonian, because screening and DFT double counting must be defined consistently.

## Python API

```python
from qeanalyzer.quantum import (
    build_integral_hamiltonian,
    write_fcidump,
    create_quantum_solver,
)

ham = build_integral_hamiltonian(
    h1,
    h2,                 # h2[p][q][r][s] = (pq|rs), chemist notation
    n_electrons=6,
    constant=ecore,
    energy_unit="Hartree",
)

write_fcidump(ham, "FCIDUMP")
result = create_quantum_solver("adapt_vqe").solve(ham)     # or "acase", "adapt_acase", ...
```

The restricted finite-Hamiltonian convention is fixed as

\[
h_{pqrs}^{(2)} = (pq|rs),
\]

with

\[
H_2=\frac12\sum_{pqrs}\sum_{\sigma\tau}(pq|rs)
 a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}.
\]

Spin-polarized (`lsda`), noncollinear, and spin-orbit QE band outputs are not silently mapped to a `2 × spatial-orbital` restricted model; the restricted active-space path raises and requires an explicit spinor/localized mapping.

## Periodic active-space semantics

QE XML k-point weights are preserved in `QEElectronicState`. Occupation and average-energy diagnostics use normalized Brillouin-zone weights, not an unweighted mean over symmetry-reduced k points.

A periodic **band index is not automatically a finite localized orbital**. Band-window selection therefore records `representation="periodic_band_indices"`; it is a selection/modeling diagnostic until an explicit Wannier or other finite-basis mapping is supplied.

Fractional DFT occupations are also not silently rounded into an FCI/ADAPT particle-number sector. FCIDUMP and finite-sector solvers require an electron count within numerical tolerance of an integer; otherwise the caller must make the sector choice explicitly.

## Feedback and outer-loop convergence

The built-in occupation, active-space, and Hubbard-U feedback policies are marked:

```text
scientific_status = experimental_heuristic
```

They are controller experiments, not a derived DFT+many-body functional.

Outer-loop convergence is fail-closed: if the RDM or residual criteria are required but unavailable, they do **not** count as passed. The third criterion is the *solver-reported residual* -- an ADAPT pool gradient, an A-CASE Ritz residual norm, or zero from the exact solver -- and each iteration records which quantity it compared (`quantum_residual_kind`). Residuals are in Hartree, the interchange unit of the `clifford_qc` boundary, while energies are in eV. Setting `require_rdm=False` or `require_gradient=False` removes that criterion from the convergence test altogether -- the quantity is still recorded for provenance, but the loop no longer closes on it.

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the core suite on Python 3.10 and 3.12, a plotting installation separately, and a dedicated integration job that checks out `gutama/clifford_qc` and runs the real ADAPT bridge test.
