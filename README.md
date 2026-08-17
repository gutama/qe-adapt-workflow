# qe-adapt-workflow

A reproducible analysis and workflow-control layer for serial Quantum ESPRESSO calculations, designed to grow toward closed-loop DFT–ADAPT-VQE workflows.

## Status

This repository is an initial scaffold. The first implementation milestone is `qeanalyzer 0.1`, focused on:

- parsing Quantum ESPRESSO inputs, text outputs, and structured XML results;
- producing versioned structured run records;
- generating human-readable diagnostics and plots;
- deterministically generating inputs for the next QE run;
- preserving provenance across serial calculations.

The longer-term architecture adds active-space construction, finite many-body Hamiltonian generation, ADAPT-VQE integration, quantum-feedback policies, and outer-loop convergence control.

## Architecture and implementation plan

The complete design is in:

- [`docs/ARCHITECTURE_IMPLEMENTATION_PLAN.md`](docs/ARCHITECTURE_IMPLEMENTATION_PLAN.md)

## Intended CLI

The initial user-facing interface is planned around commands such as:

```bash
qeanalyzer report scf.in scf.out
qeanalyzer dump scf.in scf.out -o result.json
qeanalyzer plot scf.out
qeanalyzer next scf.in scf.out --strategy scf-to-nscf -o nscf.in
```

The current scaffold only provides the package entry point and version command while the parser and workflow layers are implemented.

## Development

Create an environment and install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run tests with:

```bash
python -m unittest discover -s tests -v
```

## Project direction

The project is intentionally not positioned as another generic Quantum ESPRESSO workflow manager. Its target contribution is:

```text
QE-aware analysis
+
deterministic next-run generation
+
DFT–ADAPT-VQE coupling
```

Execution backends such as local processes, Slurm, or AiiDA can be added behind the workflow-control layer later.
