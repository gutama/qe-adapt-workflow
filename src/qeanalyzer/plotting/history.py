"""Plotting functions for multi-run workflow execution history and DAG evolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qeanalyzer.plotting.convergence import _check_matplotlib
from qeanalyzer.workflow.ledger import WorkflowLedger, WorkflowRunEntry


# Color scheme for workflow run statuses
STATUS_COLORS: dict[str, str] = {
    "converged": "#2ca02c",      # Green
    "completed": "#2ca02c",      # Green
    "not_converged": "#d62728",  # Red
    "failed": "#d62728",         # Red
    "interrupted": "#ff7f0e",    # Orange
    "running": "#1f77b4",        # Blue
    "planned": "#7f7f7f",        # Gray
    "unknown": "#9467bd",        # Purple
}


def _extract_run_metrics(
    entry: WorkflowRunEntry,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract metrics from a run entry, resolving relative files if needed."""
    metrics: dict[str, Any] = {
        "run_id": entry.run_id,
        "calculation": entry.calculation,
        "parent_run": entry.parent_run,
        "status": entry.status.lower(),
        "energy_ry": entry.energy_ry,
        "policy_applied": entry.policy_applied,
        "scf_iterations": None,
        "ionic_steps": None,
        "max_force": None,
    }

    # Attempt to load enriched info from result_json if available
    if entry.result_json:
        json_path = Path(entry.result_json)
        if not json_path.is_absolute() and base_dir is not None:
            json_path = base_dir / json_path

        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                electronic = data.get("electronic", {})
                if metrics["energy_ry"] is None:
                    metrics["energy_ry"] = electronic.get("total_energy_ry")

                convergence = data.get("convergence", {})
                scf_iters = convergence.get("scf_iterations", [])
                if scf_iters:
                    metrics["scf_iterations"] = len(scf_iters)
                ionic_steps = convergence.get("ionic_steps", [])
                if ionic_steps:
                    metrics["ionic_steps"] = len(ionic_steps)
            except Exception:
                pass

    return metrics


def plot_workflow_history(
    ledger: WorkflowLedger | str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    dpi: int = 300,
) -> Any:
    """Plot multi-run execution history and energy evolution from a workflow ledger.

    Parameters
    ----------
    ledger : WorkflowLedger or str or Path
        The workflow ledger instance or path to a workflow.json file.
    output_path : str or Path, optional
        File path to save the generated figure (e.g. .png, .pdf, .svg).
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

    base_dir = None
    if isinstance(ledger, (str, Path)):
        ledger_path = Path(ledger)
        base_dir = ledger_path.parent
        ledger_obj = WorkflowLedger.load_json(ledger_path)
    elif isinstance(ledger, WorkflowLedger):
        ledger_obj = ledger
    else:
        raise TypeError(f"Expected WorkflowLedger, str, or Path, got {type(ledger).__name__}")

    if not ledger_obj.runs:
        raise ValueError("Workflow ledger contains no runs to plot.")

    # Extract run records
    run_records = [_extract_run_metrics(r, base_dir=base_dir) for r in ledger_obj.runs]
    n_runs = len(run_records)
    indices = list(range(1, n_runs + 1))

    # Form tick labels (e.g. "1. [qe-0001]\nvc-relax")
    x_labels = [
        f"{i}. [{r['run_id']}]\n{r['calculation']}"
        for i, r in zip(indices, run_records)
    ]

    energies = [r["energy_ry"] for r in run_records]
    has_energies = any(e is not None for e in energies)

    # Determine layout based on available metrics
    if has_energies:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n_runs * 1.5), 7), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(max(8, n_runs * 1.5), 4))
        ax2 = None

    wf_title = title or f"Workflow Lineage History: {ledger_obj.workflow_id}"

    # -------------------------------------------------------------------------
    # Panel 1: Total Energy Evolution across Runs
    # -------------------------------------------------------------------------
    if has_energies:
        # Plot continuous line between valid energy points
        valid_x = [idx for idx, e in zip(indices, energies) if e is not None]
        valid_e = [e for e in energies if e is not None]

        if len(valid_x) > 1:
            ax1.plot(valid_x, valid_e, color="#1f77b4", linestyle="--", linewidth=1.5, zorder=1)

        # Plot individual points colored by status
        for idx, r, e in zip(indices, run_records, energies):
            st = r["status"]
            color = STATUS_COLORS.get(st, STATUS_COLORS["unknown"])
            if e is not None:
                ax1.scatter(
                    idx, e,
                    color=color,
                    s=100,
                    edgecolors="black",
                    linewidth=1.2,
                    zorder=3,
                )
                # Value label
                ax1.annotate(
                    f"{e:.4f} Ry",
                    (idx, e),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

        ax1.set_ylabel("Total Energy (Ry)", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.set_title(wf_title, fontsize=12, fontweight="bold")

        # ---------------------------------------------------------------------
        # Panel 2: Step-to-Step Energy Difference |E_n - E_{n-1}|
        # ---------------------------------------------------------------------
        delta_energies = []
        delta_x = []
        delta_colors = []

        for i in range(1, len(energies)):
            e_prev = energies[i - 1]
            e_curr = energies[i]
            if e_prev is not None and e_curr is not None:
                delta = abs(e_curr - e_prev)
                delta_energies.append(max(delta, 1e-12))  # prevent log 0
                delta_x.append(indices[i])
                st = run_records[i]["status"]
                delta_colors.append(STATUS_COLORS.get(st, STATUS_COLORS["unknown"]))

        if delta_energies:
            bars = ax2.bar(delta_x, delta_energies, color=delta_colors, width=0.4, edgecolor="black", linewidth=0.8)
            ax2.set_yscale("log")
            ax2.set_ylabel(r"$|\Delta E|$ (Ry)", fontsize=10, fontweight="bold")
            ax2.grid(True, linestyle=":", alpha=0.6, which="both")

            # Annotate bar values
            for bar, d in zip(bars, delta_energies):
                height = bar.get_height()
                ax2.annotate(
                    f"{d:.1e}",
                    (bar.get_x() + bar.get_width() / 2, height),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=7,
                )
        else:
            ax2.text(
                0.5, 0.5, "Single energy calculation or insufficient consecutive steps",
                ha="center", va="center", transform=ax2.transAxes, color="gray", fontsize=9,
            )
            ax2.set_ylabel(r"$|\Delta E|$ (Ry)", fontsize=10)

        # Policy transition annotations on Panel 2
        for idx, r in zip(indices, run_records):
            if r["policy_applied"]:
                ax2.annotate(
                    f"Policy: {r['policy_applied']}",
                    (idx, 0),
                    textcoords="offset points",
                    xytext=(0, -25),
                    ha="center",
                    fontsize=7,
                    fontstyle="italic",
                    color="#333333",
                )

        ax2.set_xticks(indices)
        ax2.set_xticklabels(x_labels, fontsize=9)
        ax2.set_xlabel("Workflow Calculation Sequence", fontsize=10, fontweight="bold")

    else:
        # Status timeline when no energies are recorded
        for idx, r in zip(indices, run_records):
            st = r["status"]
            color = STATUS_COLORS.get(st, STATUS_COLORS["unknown"])
            ax1.scatter(idx, 0, color=color, s=200, edgecolors="black", linewidth=1.5, zorder=3)
            status_text = st.upper().replace("_", " ")
            ax1.text(idx, 0.1, status_text, ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)
            if r["policy_applied"]:
                ax1.text(idx, -0.15, f"Policy:\n{r['policy_applied']}", ha="center", va="top", fontsize=8, fontstyle="italic")

        if len(indices) > 1:
            ax1.plot(indices, [0] * len(indices), color="#cccccc", linestyle="-", linewidth=2, zorder=1)

        ax1.set_ylim(-0.4, 0.4)
        ax1.set_yticks([])
        ax1.set_xticks(indices)
        ax1.set_xticklabels(x_labels, fontsize=9)
        ax1.set_xlabel("Workflow Calculation Sequence", fontsize=10, fontweight="bold")
        ax1.set_title(wf_title, fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.4, axis="x")

    # Legend for statuses
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=STATUS_COLORS[k], edgecolor="black", label=k.replace("_", " ").capitalize())
        for k in ["converged", "not_converged", "interrupted", "planned"]
        if any(r["status"] == k for r in run_records)
    ]
    if legend_elements:
        ax1.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi)

    return fig
