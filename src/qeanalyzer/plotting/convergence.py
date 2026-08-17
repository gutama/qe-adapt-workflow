"""Plotting functions for SCF and structural relaxation convergence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qeanalyzer.models import QERunResult


def _check_matplotlib() -> Any:
    """Ensure matplotlib is available and configure headless backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend safe for CLI / CI
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for generating convergence plots. "
            "Install it via 'pip install matplotlib' or 'pip install qe-adapt-workflow[plot]'."
        ) from exc


def plot_scf_convergence(
    result: QERunResult,
    output_path: str | Path | None = None,
    title: str | None = None,
    dpi: int = 300,
) -> Any:
    """Plot total energy and estimated SCF accuracy across iterations.

    Parameters
    ----------
    result : QERunResult
        The calculation result containing SCF iteration history.
    output_path : str or Path, optional
        File path to save the generated plot (e.g. .png, .pdf, .svg).
    title : str, optional
        Custom title for the figure.
    dpi : int, optional
        Resolution for saved image (default: 300).

    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib figure object.
    """
    plt = _check_matplotlib()

    iters = result.convergence.scf_iterations
    if not iters:
        raise ValueError("No SCF iteration history available in the provided run result.")

    iter_numbers = [it.iteration for it in iters]
    energies = [it.total_energy_ry for it in iters]
    accuracies = [it.accuracy_ry for it in iters]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # 1. Total Energy vs Iteration
    ax1.plot(iter_numbers, energies, marker="o", color="#1f77b4", linewidth=1.5, markersize=4)
    ax1.set_ylabel("Total Energy (Ry)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    calc_label = f" ({result.calculation.upper()})" if result.calculation else ""
    run_label = f"[{result.run_id}] " if result.run_id else ""
    ax1.set_title(title or f"{run_label}SCF Convergence History{calc_label}", fontsize=12, fontweight="bold")

    # 2. Estimated Accuracy vs Iteration (Log scale)
    ax2.semilogy(iter_numbers, accuracies, marker="s", color="#d62728", linewidth=1.5, markersize=4)
    ax2.set_xlabel("SCF Iteration #", fontsize=10)
    ax2.set_ylabel("Estimated Accuracy (Ry)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Add conv_thr horizontal line if available in provenance/inputs
    if result.convergence.scf_error is not None:
        ax2.axhline(
            result.convergence.scf_error,
            color="#2ca02c",
            linestyle=":",
            label=f"Target: {result.convergence.scf_error:.1e} Ry",
        )
        ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi)

    return fig


def plot_relaxation_convergence(
    result: QERunResult,
    output_path: str | Path | None = None,
    title: str | None = None,
    dpi: int = 300,
) -> Any:
    """Plot total energy, maximum atomic force, and volume across BFGS ionic relaxation steps.

    Parameters
    ----------
    result : QERunResult
        The calculation result containing ionic steps history.
    output_path : str or Path, optional
        File path to save the generated plot.
    title : str, optional
        Custom title for the figure.
    dpi : int, optional
        Resolution for saved image (default: 300).

    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib figure object.
    """
    plt = _check_matplotlib()

    steps = result.convergence.ionic_steps
    if not steps:
        raise ValueError("No ionic relaxation step history available in the provided run result.")

    step_nums = [s.step for s in steps]
    energies = [s.total_energy_ry for s in steps if s.total_energy_ry is not None]
    forces = [s.total_force for s in steps if s.total_force is not None]

    n_subplots = 2
    has_stress = any(s.stress is not None for s in steps)
    if has_stress:
        n_subplots = 3

    fig, axes = plt.subplots(n_subplots, 1, figsize=(7, 3 * n_subplots), sharex=True)
    if n_subplots == 1:
        axes = [axes]

    # 1. Total Energy vs Ionic Step
    axes[0].plot(step_nums[:len(energies)], energies, marker="o", color="#1f77b4", linewidth=1.5, markersize=5)
    axes[0].set_ylabel("Total Energy (Ry)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    run_label = f"[{result.run_id}] " if result.run_id else ""
    axes[0].set_title(title or f"{run_label}Geometry Optimization History", fontsize=12, fontweight="bold")

    # 2. Total / Max Force vs Ionic Step
    if forces:
        axes[1].plot(step_nums[:len(forces)], forces, marker="^", color="#ff7f0e", linewidth=1.5, markersize=5)
        axes[1].set_ylabel("Total Force (Ry/Bohr)", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Pressure (if vc-relax)
    if has_stress:
        pressures = [s.stress.pressure_kbar for s in steps if s.stress is not None]
        axes[2].plot(step_nums[:len(pressures)], pressures, marker="s", color="#2ca02c", linewidth=1.5, markersize=5)
        axes[2].set_ylabel("Pressure (kbar)", fontsize=10)
        axes[2].grid(True, linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("BFGS Ionic Step #", fontsize=10)
    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi)

    return fig
