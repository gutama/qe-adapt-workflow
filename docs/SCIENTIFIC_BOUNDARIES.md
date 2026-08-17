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
- finite-shot selection/measurement machinery;
- other quantum algorithms and subspace methods.

The QE project must not maintain a second scientific ADAPT implementation.

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

If `require_rdm=True`, missing or shape-incompatible 1-RDMs fail the RDM criterion. If `require_gradient=True`, an unavailable residual ADAPT gradient fails the gradient criterion. A workflow may disable either requirement explicitly, but absence is never silently converted to zero.

## 10. Source coherence

One `QERunResult` must describe one QE run. CLI discovery therefore scans one run directory, plus its immediate `*.save/data-file-schema.xml`, and rejects ambiguous multiple inputs/outputs/XML files. It does not recursively combine files from several serial calculations.
