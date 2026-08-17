# qe-adapt-workflow

A reproducible analysis, orchestration, and quantum-coupling layer for serial **Quantum ESPRESSO** calculations, designed for closed-loop **DFT–ADAPT-VQE** workflows.

---

## Key Capabilities

```text
       ┌───────────────────────────────┐
       │   Quantum ESPRESSO (DFT)      │
       │   pw.in / pw.out / XML schema │
       └──────────────┬────────────────┘
                      │ parse & validate
                      ▼
       ┌───────────────────────────────┐
       │     qeanalyzer Core Engine    │
       │  QERunResult / WorkflowLedger │
       └──────────────┬────────────────┘
                      │ active-space extraction
                      ▼
       ┌───────────────────────────────┐
       │      MaterialHamiltonian      │
       │  h1, h2 tensors / FCIDUMP     │
       └──────────────┬────────────────┘
                      │ quantum ground-state solver
                      ▼
       ┌───────────────────────────────┐
       │  ADAPT-VQE / FCI Solver       │
       │  1-RDM / Natural Occupations  │
       └──────────────┬────────────────┘
                      │ quantum feedback policy
                      ▼
       ┌───────────────────────────────┐
       │   Outer-Loop Self-Consistency │
       │  |ΔE| < ε_E, ||Δγ|| < ε_γ     │
       └───────────────────────────────┘
```

1. **Robust I/O & Parsing**: Full Fortran namelist input parser/writer (`pw.in`), text output parser (`pw.out`), and schema-aware XML parser (`data-file-schema.xml`).
2. **Canonical Data Representation**: Complete data model (`QERunResult`) with electronic structure, energetics, convergence history, geometry, stress/forces, and diagnostics.
3. **Reporting & Visualizations**: Human-readable text and Markdown reports, JSON Schema exports, publication-grade SCF and relaxation convergence plots, and DAG workflow history plots.
4. **State Machine & Policies**: Deterministic calculation transitions (`relax_to_scf`, `scf_to_nscf`) and automated recovery (`interrupted`, `unconverged_scf`).
5. **DAG Workflow Ledger**: Immutable JSON workflow run ledger (`WorkflowLedger`) tracking calculation lineage, parents, applied policies, and DAG consistency.
6. **Active-Space Selection**: Multiple selection strategies (`energy_window`, `band_index`, `occupation`, `explicit`).
7. **Hamiltonian Modeling & FCIDUMP**: Constructs 1-body and 2-body electronic tensors, parameterized Hubbard lattice models ($t, U, V, J$), and exports/imports standard `FCIDUMP` format.
8. **Quantum Solver Bridge**: Includes exact Full Configuration Interaction (FCI) reference solver, iterative ADAPT-VQE gradient optimizer, and 1-RDM / natural orbital occupation extractors.
9. **Quantum Feedback Policies**: Translates quantum correlated 1-RDMs into deterministic input modifications (`OccupationFeedbackPolicy`, `ActiveSpaceFeedbackPolicy`, `HubbardUFeedbackPolicy`).
10. **Outer-Loop Convergence & Orchestrator**: Self-consistent multi-criteria outer-loop convergence checker ($|\Delta E_n| < \epsilon_E$, $\|\gamma_n - \gamma_{n-1}\|_F < \epsilon_\gamma$, $\max |g_k| < \epsilon_g$) and automated controller (`OuterLoopController`).
11. **Execution Abstraction**: Decoupled runner layer supporting synchronous/asynchronous `LocalRunner` (serial & MPI) and batch `SlurmRunner` (batch script rendering and job polling).

---

## Installation

Create a virtual environment and install in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run test suite:

```bash
python3 -m unittest discover -s tests -v
```

---

## Command-Line Interface (CLI)

The `qeanalyzer` command-line tool provides a unified entrypoint for analysis, report generation, workflow progression, plotting, and FCIDUMP export.

### 1. Summary Report

```bash
# Generate plain text report
qeanalyzer report pw.in pw.out data-file-schema.xml

# Generate GitHub-flavored Markdown report
qeanalyzer report run_dir/ --format markdown -o report.md
```

### 2. JSON Schema Dump

```bash
qeanalyzer dump pw.out data-file-schema.xml -o result.json
```

### 3. Convergence & History Plots

```bash
# Plot SCF or relaxation convergence
qeanalyzer plot pw.out -o convergence.png --dpi 300

# Plot full workflow DAG history from ledger
qeanalyzer plot-history workflow.json -o history.png
```

### 4. Deterministic Next-Step Input Planning

```bash
# Automatically transition a converged relax calculation to SCF
qeanalyzer next relax/ -o 002_scf/ --ledger workflow.json

# Explicitly transition SCF to NSCF with expanded bands
qeanalyzer next scf/ --policy scf_to_nscf -o 003_nscf/ --ledger workflow.json
```

### 5. Workflow Ledger Inspection & Validation

```bash
# View calculation execution history table
qeanalyzer history workflow.json

# Validate DAG lineage and ledger integrity
qeanalyzer validate workflow.json
```

### 6. Active-Space Hamiltonian & FCIDUMP Export

```bash
# Export active space Hamiltonian from QE output to standard FCIDUMP format
qeanalyzer export-fcidump pw.out data-file-schema.xml \
    -o model.fcidump \
    --active-method energy_window \
    --emin -3.0 --emax 3.0 \
    -u 2.5 -v 0.5
```

---

## Python API & Examples

Runnable standalone example scripts are provided in [`examples/`](examples/):

- [`examples/01_parse_and_report.py`](examples/01_parse_and_report.py): Parse calculation files, extract `QERunResult`, and produce reports and JSON schemas.
- [`examples/02_workflow_progression.py`](examples/02_workflow_progression.py): Plan deterministic `relax` $\to$ `scf` $\to$ `nscf` calculations and record execution in `WorkflowLedger`.
- [`examples/03_dft_to_quantum_solver.py`](examples/03_dft_to_quantum_solver.py): Extract active space, build `MaterialHamiltonian`, export `FCIDUMP`, solve with Exact Diagonalization & ADAPT-VQE, and evaluate quantum feedback.
- [`examples/04_outer_loop_cycle.py`](examples/04_outer_loop_cycle.py): Coordinate multi-step self-consistent DFT-ADAPT outer loops with multi-criteria convergence checking.

To run all examples:

```bash
python3 examples/01_parse_and_report.py
python3 examples/02_workflow_progression.py
python3 examples/03_dft_to_quantum_solver.py
python3 examples/04_outer_loop_cycle.py
```

---

## Architecture Overview

Refer to the complete design specification in [`docs/ARCHITECTURE_IMPLEMENTATION_PLAN.md`](docs/ARCHITECTURE_IMPLEMENTATION_PLAN.md) for full architectural details, data flow diagrams, and mathematical formulation.
