# QE-ADAPT Architecture and Implementation Plan

## 1. Project Goal

`qe-adapt-workflow` is intended to be more than a Quantum ESPRESSO output parser. It should become a lightweight, reproducible **closed-loop workflow system** for serial DFT–ADAPT-VQE calculations.

Core loop:

```text
QE run n
   ↓
parse + validate + analyze
   ↓
structured result
   ↓
deterministic next-run policy
   ↓
generate QE input for run n+1
   ↓
optional active-space / Hamiltonian construction
   ↓
ADAPT-VQE
   ↓
quantum feedback
   ↓
next QE run
```

The long-term objective is:

> Build a reproducible controller for serial DFT–ADAPT-VQE calculations in which the state produced by each electronic-structure or quantum-solver stage deterministically determines the next calculation.

The project should reuse the reproducibility philosophy of `gutama/adapt_vqe_enc_meas`: calculation → structured result → deterministic summary → validation → derived reports/plots/data. For Quantum ESPRESSO, that chain is extended into a feedback loop because outputs from one run determine later inputs.

---

## 2. High-Level Architecture

```text
                 ┌───────────────────┐
                 │ QE input run n    │
                 │ pw.in             │
                 └─────────┬─────────┘
                           │
                           ▼
                    ┌────────────┐
                    │ pw.x       │
                    └─────┬──────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
          pw.out                prefix.save/
                                      │
                              data-file-schema.xml
             │                         │
             └────────────┬────────────┘
                          ▼
                 ┌──────────────────┐
                 │ qeanalyzer       │
                 │ parser           │
                 │ validators       │
                 │ diagnostics      │
                 │ plots            │
                 └────────┬─────────┘
                          │
                     result.json
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       report.md       figures/      next-run policy
                                             │
                                             ▼
                                      pw_next.in
                                             │
                                             ▼
                                         QE run n+1
```

QE structured XML should be treated as an important source of truth, while plain-text output parsing provides diagnostics, warnings, iteration histories, timing, and error information.

---

## 3. Recommended Package Structure

Do not build a monolithic `qeanalyzer.py`. Keep the CLI simple while the implementation is modular.

```text
qe-adapt-workflow/
├── pyproject.toml
├── README.md
├── docs/
│   └── ARCHITECTURE_IMPLEMENTATION_PLAN.md
├── src/qeanalyzer/
│   ├── __init__.py
│   ├── cli.py
│   ├── io/
│   │   ├── pw_input.py
│   │   ├── pw_output.py
│   │   ├── qe_xml.py
│   │   ├── bands.py
│   │   ├── dos.py
│   │   └── projwfc.py
│   ├── models/
│   │   ├── run.py
│   │   ├── structure.py
│   │   ├── electronic.py
│   │   └── provenance.py
│   ├── analysis/
│   │   ├── convergence.py
│   │   ├── scf.py
│   │   ├── structure.py
│   │   ├── bands.py
│   │   ├── magnetism.py
│   │   └── warnings.py
│   ├── plotting/
│   │   ├── convergence.py
│   │   ├── bands.py
│   │   ├── dos.py
│   │   └── structure.py
│   ├── workflow/
│   │   ├── state.py
│   │   ├── policy.py
│   │   ├── next_input.py
│   │   └── runner.py
│   ├── quantum/
│   │   ├── active_space.py
│   │   ├── hamiltonian.py
│   │   ├── fcidump.py
│   │   └── adapt_bridge.py
│   └── report/
│       ├── text.py
│       └── json.py
├── schemas/
│   ├── qe_run.schema.json
│   ├── adapt_run.schema.json
│   └── workflow.schema.json
├── tests/
│   ├── fixtures/
│   ├── test_pw_input.py
│   ├── test_pw_output.py
│   ├── test_qe_xml.py
│   ├── test_next_input.py
│   ├── test_workflow.py
│   └── test_quantum_bridge.py
└── examples/
    ├── silicon/
    ├── molecule/
    └── dft_adapt/
```

---

## 4. CLI Design

Initial commands:

```bash
qeanalyzer parse scf.out
qeanalyzer report scf.in scf.out
qeanalyzer dump scf.in scf.out -o result.json
qeanalyzer plot scf.out --what scf
qeanalyzer next scf.in scf.out --strategy scf-to-nscf -o nscf.in
```

Later:

```bash
qeanalyzer workflow dft-adapt workflow.yaml
qeanalyzer validate run/
qeanalyzer history workflow.json
qeanalyzer plot history workflow.json
qeanalyzer export fcidump run/
qeanalyzer inspect prefix.save/
```

---

## 5. Parser Layer

### 5.1 Structured XML parser

Primary target:

```text
prefix.save/data-file-schema.xml
```

The parser should extract structured quantities into a canonical result model, for example:

```python
QERunResult(
    calculation="scf",
    converged=True,
    total_energy_ry=-114.3271,
    fermi_energy_ev=5.12,
    n_electrons=8.0,
    n_bands=20,
    kpoints=...,
    eigenvalues=...,
    occupations=...,
    forces=...,
    stress=...,
    structure=...,
)
```

The XML parser should be version-aware. Unknown schema versions should produce an explicit warning or unsupported-version error rather than silently changing semantics.

### 5.2 Text-output parser

`pw.out` remains essential for workflow diagnostics. Parse:

```text
QE version
calculation type
SCF iteration history
total energy by iteration
estimated SCF accuracy
convergence status
Davidson/CG diagnostics
warnings and error blocks
CPU/wall time
memory estimates
Fermi/HOMO information
forces
stress
magnetization
```

Prefer structured events over loose dictionaries, e.g.:

```python
SCFIteration(
    iteration=7,
    total_energy_ry=-114.327012,
    estimated_accuracy_ry=2.3e-8,
)
```

---

## 6. Canonical Result Model

All calculations should produce a versioned structured record.

```json
{
  "schema_version": "1.0",
  "run_id": "qe-0003",
  "parent_run": "qe-0002",
  "engine": {
    "name": "Quantum ESPRESSO",
    "executable": "pw.x",
    "version": "7.5"
  },
  "calculation": "scf",
  "status": {
    "completed": true,
    "scf_converged": true
  },
  "electronic": {
    "total_energy_ry": -114.3271,
    "fermi_energy_ev": 4.81,
    "n_electrons": 12,
    "n_bands": 24
  },
  "provenance": {
    "input_sha256": "...",
    "pseudo_sha256": {"Si": "..."}
  }
}
```

Recommended classes:

```text
QERunResult
QERunStatus
QEElectronicState
QEStructure
QEConvergenceHistory
QEProvenance
QEWarning
QEError
```

---

## 7. Provenance Requirements

Each run should record enough information to reproduce and audit it:

```text
QE version
qeanalyzer version
input hash
pseudopotential hashes
prefix/outdir
calculation type
k-point mesh
ecutwfc/ecutrho
occupations/smearing
number of bands
MPI ranks / OpenMP threads
hostname or scheduler job ID when available
parent run
policy that generated the input
policy version
```

Every derived input should be traceable to the exact prior state and rule that generated it.

---

## 8. Human-Readable Reports

`qeanalyzer report` should combine factual parsed quantities, derived diagnostics, and the recommended next action, clearly labeling which is which.

Example:

```text
Quantum ESPRESSO Run Analysis
=============================

Run
---
QE version          : 7.5
Calculation         : scf
Status              : CONVERGED
SCF iterations      : 11

Electronic structure
--------------------
Total energy         : -15.842731 Ry
Fermi energy         : 6.214 eV
Band gap             : 1.18 eV

Convergence
-----------
Final SCF accuracy   : 2.1e-10 Ry
Energy Δ             : 3.4e-9 Ry

Diagnostics
-----------
✓ SCF converged
✓ No NaN/Inf
⚠ k-point convergence not established

Recommended next run
--------------------
NSCF calculation

Generated:
 runs/002_nscf/pw.in
```

---

## 9. Analysis Layer

Initial analysis modules should cover:

### SCF
- convergence status;
- iteration count;
- total-energy history;
- estimated accuracy;
- charge-density/mixing behavior;
- occupied-highest-band warnings.

### Structure
- max/RMS force;
- stress;
- cell-volume evolution;
- geometry convergence;
- final structure extraction.

### Electronic structure
- Fermi level;
- HOMO/LUMO when meaningful;
- band gap;
- metallic/insulating heuristic;
- occupations;
- magnetization;
- number of empty bands;
- k-point coverage.

### Run health
- normal completion;
- QE error blocks;
- time-limit interruption;
- NaN/Inf;
- memory or diagonalization failures;
- incomplete calculations.

---

## 10. Plotting Layer

Initial plots:

```text
SCF energy vs iteration
SCF accuracy vs iteration
forces vs ionic step
cell volume vs ionic step
stress vs ionic step
band structure
DOS / projected DOS
magnetization
```

Serial DFT–ADAPT history plots should later include:

```text
QE energy vs outer iteration
ADAPT energy vs outer iteration
DFT–ADAPT correction
SCF iterations
active-space size
number of ADAPT operators
orbital occupations
band gap
max force
RDM change
```

Useful outer-loop quantities include:

\[
\Delta E_n = |E_n-E_{n-1}|,
\qquad
\Delta \rho_n = \|\rho_n-\rho_{n-1}\|,
\qquad
\Delta \gamma_n = \|\gamma_n-\gamma_{n-1}\|_F.
\]

---

## 11. Deterministic Next-Input Generation

This is the central architectural feature.

Use a policy interface rather than a large hard-coded `if/else` tree:

```python
class NextRunPolicy:
    def evaluate(
        self,
        previous_input,
        previous_result,
        workflow_state,
    ) -> NextRunDecision:
        ...
```

Example:

```python
class SCFToNSCFPolicy:
    def evaluate(self, result):
        assert result.status.scf_converged
        nbnd = max(
            result.electronic.n_bands,
            ceil(result.electronic.n_occupied * 1.5),
        )
        return QEInputPatch(
            CONTROL={"calculation": "nscf"},
            SYSTEM={"nbnd": nbnd},
        )
```

A next-run decision should contain the policy name/version, changed parameters, reason, parent run, and generated input hash.

---

## 12. Restart vs Next Run

Keep these semantics separate.

```text
RESTART
same calculation was interrupted
        ↓
continue that run
```

versus:

```text
NEXT RUN
previous calculation completed
        ↓
start a new calculation using its results
```

For example:

```text
vc-relax → final geometry → SCF
```

is a new run, not a restart. Model these separately, e.g. `RestartDecision` and `NextRunDecision`, and enforce the distinction in tests.

---

## 13. Initial QE Workflow State Machine

```text
STRUCTURE
    │
    ▼
VC-RELAX
    │ final structure
    ▼
SCF
    │ converged density
    ▼
NSCF
    │
    ├──────────────┐
    ▼              ▼
 BANDS          PROJWFC
                   │
                   ▼
               WANNIER90
```

Initial policy transitions:

```text
relax → scf
vc-relax → scf
scf → nscf
scf → bands
scf → projwfc
```

---

## 14. Workflow Ledger

Use a versioned `workflow.json` to record serial provenance.

```json
{
  "schema_version": "1.0",
  "workflow_id": "silicon-001",
  "runs": [
    {"run_id": "qe-0001", "type": "qe", "calculation": "vc-relax"},
    {"run_id": "qe-0002", "type": "qe", "calculation": "scf", "parent_run": "qe-0001"},
    {"run_id": "qe-0003", "type": "qe", "calculation": "nscf", "parent_run": "qe-0002"}
  ]
}
```

This supports:

```bash
qeanalyzer history workflow.json
qeanalyzer plot history workflow.json
qeanalyzer validate workflow.json
```

---

## 15. DFT–ADAPT-VQE Architecture

Long-term loop:

```text
┌────────────────────────────────────────────┐
│                                            │
▼                                            │
QE SCF                                       │
│                                            │
├─ density                                   │
├─ Kohn–Sham orbitals                        │
├─ eigenvalues                               │
└─ occupations                               │
       │                                     │
       ▼                                     │
Active-space selector                        │
       │                                     │
       ▼                                     │
Hamiltonian builder                          │
       │                                     │
       ├── FCIDUMP (optional)                │
       ▼                                     │
Fermion Hamiltonian                          │
       │                                     │
       ▼                                     │
ADAPT-VQE                                    │
       │                                     │
       ├─ energy                             │
       ├─ selected operators                 │
       ├─ state                              │
       └─ RDM / observables                  │
                 │                           │
                 ▼                           │
           feedback policy                   │
                 │                           │
                 ▼                           │
          next QE input ─────────────────────┘
```

The intended contribution is therefore:

```text
QE-aware analysis
+
deterministic next-run generation
+
DFT–ADAPT-VQE coupling
```

---

## 16. Active-Space Layer

The active-space selector should be independent of the feedback policy. Possible deterministic selection criteria:

```text
energy window around the Fermi level
band-index range
orbital projection weight
Wannier character
occupation threshold
symmetry
explicit user orbital list
```

Record the active-space selection method, thresholds, electron count, orbital ordering, and resulting spin-orbital count in the workflow ledger.

---

## 17. Hamiltonian Construction

For periodic materials, FCIDUMP should be an interchange format rather than the internal source of truth.

Preferred path:

```text
QE
 ↓
Wannier90
 ↓
interaction construction / cRPA
 ↓
MaterialHamiltonian
 ↓
FermionOperator
```

Optional export:

```text
MaterialHamiltonian → FCIDUMP
```

Suggested native representation:

```python
MaterialHamiltonian(
    orbitals=...,
    electrons=...,
    lattice=...,
    hopping_t=...,
    onsite_U=...,
    intersite_V=...,
    exchange_J=...,
    constant=...,
)
```

When full `h[p,q]` and `g[p,q,r,s]` tensors are genuinely available, an FCIDUMP exporter can be used directly.

---

## 18. Quantum Solver Bridge

Expose a stable solver interface:

```python
class QuantumSolver:
    def solve(
        self,
        hamiltonian,
        active_space,
        initial_state=None,
    ) -> QuantumRunResult:
        ...
```

A quantum run result should record at least energy, active-space metadata, selected ADAPT operators, convergence metrics, and available RDM/observable information.

The existing `gutama/adapt_vqe_enc_meas` project should remain a separate quantum-algorithm/reproducibility package. `qe-adapt-workflow` should call or interoperate with it rather than merge the repositories.

---

## 19. Quantum Feedback Policy

Do not prematurely hard-code how ADAPT-VQE modifies the next DFT calculation. Keep the feedback rule behind an interface:

```python
class QuantumFeedbackPolicy:
    def update(
        self,
        qe_result,
        quantum_result,
        workflow_state,
    ) -> QEInputPatch:
        ...
```

Potential future implementations:

```text
OccupationFeedbackPolicy
ActiveSpaceFeedbackPolicy
GeometryFeedbackPolicy
PotentialFeedbackPolicy
DensityMatrixFeedbackPolicy
```

The correct physical feedback rule is part of the scientific DFT–ADAPT-VQE method and may evolve independently of parsing/orchestration.

---

## 20. Outer-Loop Convergence

DFT–ADAPT convergence must be distinct from SCF convergence. Candidate criteria include:

\[
|E_n-E_{n-1}| < \epsilon_E,
\qquad
\|\gamma_n-\gamma_{n-1}\|_F < \epsilon_\gamma,
\qquad
\|\rho_n-\rho_{n-1}\| < \epsilon_\rho,
\qquad
\max_k |g_k| < \epsilon_g.
\]

A workflow report should state whether the outer loop converged, which criteria passed, and why another QE run is or is not required.

---

## 21. Runner Layer

Do not couple scientific workflow logic to a single execution environment.

```python
class Runner:
    def submit(self, run_spec):
        ...

    def status(self, run_id):
        ...
```

Initial implementations:

```text
LocalRunner
SlurmRunner
```

Optional later backend:

```text
AiiDARunner
```

---

## 22. Relationship to AiiDA

The project should not rebuild generic execution, restart handling, scheduler integration, and provenance already provided by mature systems such as AiiDA.

Positioning:

```text
qeanalyzer ≠ AiiDA replacement
```

Instead:

```text
qeanalyzer
    ├── LocalRunner
    ├── SlurmRunner
    └── AiiDARunner
```

`qeanalyzer` owns domain-specific analysis and DFT–ADAPT decision logic; AiiDA may later serve as an execution/provenance backend.

---

## 23. Validation Strategy

### Parser fixtures

Include successful and failed SCF, relax, vc-relax, NSCF, metal, insulator, spin-polarized, warning-containing, and interrupted calculations.

### Input round-trip tests

```text
QE input → parse → modify → write → parse again
```

Assert semantic equality.

### Next-run policy tests

For every policy:

```text
previous input + previous result + workflow state → expected next input
```

### Corruption tests

Test missing XML, truncated stdout, invalid numeric values, NaN, inconsistent electron counts, missing pseudopotentials, failed convergence, and malformed workflow state.

### Scientific invariants

At minimum:

```text
completed SCF must not be treated as restart
next run must reference the intended parent state
generated input must preserve pseudopotential provenance
active-space indexing must remain stable
Hamiltonian must be Hermitian
electron number must agree across QE and quantum layers
```

---

## 24. Determinism Requirements

For identical previous input, output, policy version, and configuration, the generated `result.json`, report, and next input should be identical except for explicitly excluded metadata such as timestamps.

Avoid hidden nondeterminism from unordered collections, filesystem ordering, implicit random seeds, raw floating-point comparisons, or version-dependent defaults.

Canonicalize at least:

```text
orbital ordering
k-point ordering
atomic ordering
JSON key ordering
float formatting
warning ordering
input namelist ordering
```

---

## 25. Configuration

A workflow can be defined in YAML, for example:

```yaml
workflow:
  name: dft-adapt-demo
  schema_version: 1

qe:
  executable: pw.x
  convergence:
    scf_accuracy_ry: 1.0e-10

active_space:
  method: energy_window
  emin_ev: -3.0
  emax_ev: 2.0

quantum:
  solver: adapt_vqe

outer_convergence:
  energy_ev: 1.0e-4
  rdm_frobenius: 1.0e-3

runner:
  backend: local
```

The fully resolved configuration should be stored in the workflow record.

---

## 26. Minimum Viable Product

### M1 — QE Analyzer

```bash
qeanalyzer report pw.in pw.out
qeanalyzer dump pw.in pw.out -o result.json
qeanalyzer plot pw.out
```

Initial support:

```text
pw.x SCF
pw.x relax
pw.x vc-relax
pw.x NSCF
```

Outputs include energy, convergence, structure, forces, stress, Fermi level, eigenvalues, occupations, magnetization, timing, warnings, and errors.

### M2 — Deterministic next-run generator

```bash
qeanalyzer next run1/
```

Initial transitions:

```text
relax → scf
vc-relax → scf
scf → nscf
scf → bands
scf → projwfc
```

### M3 — Workflow ledger

Introduce `workflow.json`, history/validation commands, and deterministic provenance across a complete chain such as:

```text
structure → vc-relax → scf → nscf → bands/projwfc
```

### M4 — DFT–ADAPT bridge

```text
QE
 ↓
active-space construction
 ↓
Hamiltonian construction
 ↓
FCIDUMP / FermionOperator
 ↓
ADAPT-VQE
 ↓
feedback policy
 ↓
next QE input
```

The first quantum-integrated validation target should be small enough for exact diagonalization.

---

## 27. Suggested Development Sequence

```text
1. repository scaffold
2. QE input parser/writer
3. pw.out parser
4. XML parser
5. canonical QERunResult
6. JSON serialization
7. qeanalyzer report
8. SCF convergence plots
9. validators
10. next-run policy interface
11. SCF → NSCF
12. relax / vc-relax → SCF
13. workflow ledger
14. history plots
15. active-space interface
16. Hamiltonian interface
17. FCIDUMP export
18. ADAPT bridge
19. quantum feedback interface
20. outer-loop convergence
21. Slurm runner
22. optional AiiDA backend
```

---

## 28. Recommended First End-to-End Demonstration

Progress in stages:

```text
Stage A:
Si or another simple QE SCF
→ parser
→ report
→ plots
→ NSCF generation

Stage B:
small molecule or tiny periodic model
→ active-space selection
→ finite Hamiltonian
→ exact diagonalization

Stage C:
same Hamiltonian
→ ADAPT-VQE
→ compare against exact result

Stage D:
one well-defined feedback rule
→ regenerate next QE input

Stage E:
serial QE–ADAPT iterations
→ convergence report
```

Do not start with a difficult correlated solid before the full chain is validated.

---

## 29. Scientific Positioning

Avoid positioning the project as merely:

> A Python parser for Quantum ESPRESSO.

Also avoid claiming it is a new generic QE workflow manager.

Stronger positioning:

> A reproducible analysis and control layer for serial Quantum ESPRESSO calculations, with deterministic next-run generation and explicit support for closed-loop DFT–ADAPT-VQE workflows.

Possible paper/software framing:

> **QE-ADAPT: A Reproducible Closed-Loop Workflow for Quantum ESPRESSO and ADAPT-VQE**

Potential contributions:

1. structured multi-source QE parsing;
2. deterministic result schemas;
3. automated diagnostics;
4. deterministic next-input policies;
5. serial workflow provenance;
6. active-space handoff;
7. material Hamiltonian representation;
8. ADAPT-VQE coupling;
9. explicit quantum-to-DFT feedback policies;
10. outer-loop convergence analysis.

---

## 30. Design Principles

1. **Parsing and physics are separate:** parser ≠ analysis ≠ workflow decision.
2. **Every decision is reproducible:** previous state + policy = next state.
3. **Preserve provenance:** every derived input is traceable to its source calculation, parsed result, policy, and configuration.
4. **FCIDUMP is an interchange format:** prefer a native `MaterialHamiltonian` representation for periodic/downfolded systems.
5. **Do not prematurely freeze DFT–ADAPT feedback physics:** keep it behind an interface.
6. **Restart and next-run semantics remain distinct.**
7. **Lightweight first, scalable later:** Python + QE + `qeanalyzer` first; Slurm/AiiDA/HPC integration later.

---

## 31. Immediate First Milestone

Target `qeanalyzer 0.1` with:

```bash
qeanalyzer report scf.in scf.out
qeanalyzer dump scf.in scf.out -o result.json
qeanalyzer plot scf.out
qeanalyzer next scf.in scf.out --strategy scf-to-nscf -o nscf.in
```

Supported calculations:

```text
SCF
NSCF
relax
vc-relax
```

Required tests:

```text
successful parsing
failed SCF
interrupted run
input round-trip
XML/text consistency
SCF → NSCF deterministic generation
vc-relax → SCF final-structure propagation
restart-vs-next-run invariant
```

This is large enough to demonstrate the supervisor's concept while remaining small enough for rigorous validation.

---

## 32. Long-Term Vision

```text
structure
   ↓
QE relaxation
   ↓
QE SCF
   ↓
electronic analysis
   ↓
active-space selection
   ↓
Wannier / Hamiltonian construction
   ↓
ADAPT-VQE
   ↓
quantum observables / RDM
   ↓
feedback policy
   ↓
next QE input
   ↓
repeat
   ↓
outer-loop convergence
   ↓
report + plots + provenance
```

At maturity, `qeanalyzer` is not only an analyzer but a domain-specific, reproducible **DFT–quantum workflow controller**.

The architectural objective is:

\[
\boxed{
\text{QE analysis}
+
\text{deterministic workflow control}
+
\text{DFT–ADAPT-VQE coupling}
}
\]

rather than simply a QE output parser.
