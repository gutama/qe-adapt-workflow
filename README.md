# qe-adapt-workflow

A reproducible analysis and workflow-control layer for serial **Quantum ESPRESSO** calculations, with an explicit bridge to correlated quantum solvers.

The repository is intentionally split from [`gutama/clifford_qc`](https://github.com/gutama/clifford_qc):

- **qe-adapt-workflow owns** QE parsing, diagnostics, run provenance, next-input generation, active-space *selection metadata*, finite-Hamiltonian interchange, and outer-loop orchestration.
- **clifford_qc owns** ADAPT-VQE, Pauli/Clifford operator algebra, measurement/selection machinery, and related quantum algorithms.

This avoids maintaining a second scientific ADAPT implementation here.

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
              gutama/clifford_qc                     │
              real ADAPT-VQE                         │
                     │                               │
                     ▼                               │
            energy / state / 1-RDM                   │
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

For **real ADAPT-VQE**, install the sibling `clifford_qc` project in the same environment. The chemistry/openfermion extra supplies the excitation-pool adapter used by this repository:

```bash
pip install -e '../clifford_qc[openfermion]'
```

If `clifford_qc` is not installed, `create_quantum_solver("adapt_vqe")` fails loudly. A separate `SimulatedADAPTVQESolver` exists only for workflow plumbing tests and is explicitly marked non-scientific.

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
    CliffordQCADAPTSolver,
)

ham = build_integral_hamiltonian(
    h1,
    h2,                 # h2[p][q][r][s] = (pq|rs), chemist notation
    n_electrons=6,
    constant=ecore,
    energy_unit="Hartree",
)

write_fcidump(ham, "FCIDUMP")
result = CliffordQCADAPTSolver().solve(ham)
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

Outer-loop convergence is fail-closed: if RDM or ADAPT-gradient criteria are required but unavailable, they do **not** count as passed. Set `require_rdm=False` or `require_gradient=False` only when that omission is an explicit workflow choice.

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the core suite on Python 3.10 and 3.12, a plotting installation separately, and a dedicated integration job that checks out `gutama/clifford_qc` and runs the real ADAPT bridge test.
