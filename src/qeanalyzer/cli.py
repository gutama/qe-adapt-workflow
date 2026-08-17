"""Command-line entry point for qeanalyzer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .io import PWInput, PWOutput, QEXMLOutput, read_pw_input, read_pw_output, read_qe_xml
from .models import build_run_result
from .plotting import plot_relaxation_convergence, plot_scf_convergence, plot_workflow_history
from .report import dump_result_json, generate_text_report, save_result_json, save_text_report
from .workflow import WorkflowLedger, default_registry, plan_next_calculation


def _detect_and_load_sources(paths: list[str]) -> tuple[PWInput | None, PWOutput | None, QEXMLOutput | None, str | None]:
    """Detect and parse inputs, outputs, and XML files from specified file or directory paths."""
    pw_in: PWInput | None = None
    pw_out: PWOutput | None = None
    qe_xml: QEXMLOutput | None = None
    input_text: str | None = None

    all_files: list[Path] = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_dir():
            # Scan directory for relevant files
            for f in p.glob("**/*"):
                if f.is_file():
                    all_files.append(f)
        elif p.is_file():
            all_files.append(p)

    for f in all_files:
        name = f.name.lower()
        if name.endswith(".xml") or name == "data-file-schema.xml":
            if qe_xml is None:
                try:
                    qe_xml = read_qe_xml(f)
                except Exception:
                    pass
        elif name.endswith(".out") or name.endswith(".log"):
            if pw_out is None:
                try:
                    pw_out = read_pw_output(f)
                except Exception:
                    pass
        elif name.endswith(".in") or name.endswith(".pwi"):
            if pw_in is None:
                try:
                    input_text = f.read_text()
                    pw_in = read_pw_input(f)
                except Exception:
                    pass

    return pw_in, pw_out, qe_xml, input_text


def cmd_dump(args: argparse.Namespace) -> int:
    """Handle `qeanalyzer dump` subcommand."""
    pw_in, pw_out, qe_xml, input_text = _detect_and_load_sources(args.paths)

    if not pw_in and not pw_out and not qe_xml:
        sys.stderr.write("Error: No valid Quantum ESPRESSO input (.in), output (.out), or XML (.xml) files found.\n")
        return 1

    result = build_run_result(
        pw_in=pw_in,
        pw_out=pw_out,
        qe_xml=qe_xml,
        run_id=args.run_id,
        parent_run=args.parent_run,
        input_text=input_text,
    )

    if args.output:
        save_result_json(result, args.output, indent=args.indent)
    else:
        sys.stdout.write(dump_result_json(result, indent=args.indent))

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Handle `qeanalyzer report` subcommand."""
    pw_in, pw_out, qe_xml, input_text = _detect_and_load_sources(args.paths)

    if not pw_in and not pw_out and not qe_xml:
        sys.stderr.write("Error: No valid Quantum ESPRESSO input (.in), output (.out), or XML (.xml) files found.\n")
        return 1

    result = build_run_result(
        pw_in=pw_in,
        pw_out=pw_out,
        qe_xml=qe_xml,
        run_id=args.run_id,
        parent_run=args.parent_run,
        input_text=input_text,
    )

    report_text = generate_text_report(result, markdown=args.markdown)

    if args.output:
        save_text_report(result, args.output, markdown=args.markdown)
    else:
        sys.stdout.write(report_text)

    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Handle `qeanalyzer plot` subcommand."""
    # Check if this is a workflow history plot
    if getattr(args, "what", None) == "history" or (len(args.paths) == 1 and Path(args.paths[0]).suffix.lower() == ".json"):
        potential_path = Path(args.paths[0])
        if potential_path.is_file():
            try:
                ledger = WorkflowLedger.load_json(potential_path)
                output_path = args.output or "workflow_history.png"
                plot_workflow_history(ledger, output_path=output_path, title=args.title, dpi=args.dpi)
                sys.stdout.write(f"Saved workflow history plot to {output_path}\n")
                return 0
            except Exception as exc:
                if getattr(args, "what", None) == "history":
                    sys.stderr.write(f"Error generating workflow history plot: {exc}\n")
                    return 1

    pw_in, pw_out, qe_xml, input_text = _detect_and_load_sources(args.paths)

    if not pw_in and not pw_out and not qe_xml:
        sys.stderr.write("Error: No valid Quantum ESPRESSO input (.in), output (.out), XML (.xml), or ledger (.json) files found.\n")
        return 1

    result = build_run_result(
        pw_in=pw_in,
        pw_out=pw_out,
        qe_xml=qe_xml,
        run_id=args.run_id,
        parent_run=args.parent_run,
        input_text=input_text,
    )

    output_path = args.output
    what = args.what

    if what is None:
        if (result.calculation in ("relax", "vc-relax") or len(result.convergence.ionic_steps) > 1 or result.status.opt_converged):
            what = "relax"
        else:
            what = "scf"

    if output_path is None:
        output_path = f"{what}_convergence.png"

    try:
        if what == "scf":
            plot_scf_convergence(result, output_path=output_path, title=args.title, dpi=args.dpi)
        elif what in ("relax", "vc-relax"):
            plot_relaxation_convergence(result, output_path=output_path, title=args.title, dpi=args.dpi)
        else:
            sys.stderr.write(f"Error: Unknown plot target '{what}'. Choose 'scf', 'relax', or 'history'.\n")
            return 1
    except Exception as exc:
        sys.stderr.write(f"Error generating plot: {exc}\n")
        return 1

    sys.stdout.write(f"Saved {what} convergence plot to {output_path}\n")
    return 0


def cmd_plot_history(args: argparse.Namespace) -> int:
    """Handle `qeanalyzer plot-history` subcommand."""
    l_path = Path(args.ledger)
    if not l_path.exists():
        sys.stderr.write(f"Error: Ledger file not found at '{args.ledger}'\n")
        return 1

    try:
        ledger = WorkflowLedger.load_json(l_path)
        output_path = args.output or "workflow_history.png"
        plot_workflow_history(ledger, output_path=output_path, title=args.title, dpi=args.dpi)
        sys.stdout.write(f"Saved workflow history plot to {output_path}\n")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error generating workflow history plot: {exc}\n")
        return 1


def cmd_next(args: argparse.Namespace) -> int:
    """Handle `qeanalyzer next` subcommand."""
    pw_in, pw_out, qe_xml, input_text = _detect_and_load_sources(args.paths)

    if not pw_in and not pw_out and not qe_xml:
        sys.stderr.write("Error: No valid Quantum ESPRESSO input (.in), output (.out), or XML (.xml) files found.\n")
        return 1

    result = build_run_result(
        pw_in=pw_in,
        pw_out=pw_out,
        qe_xml=qe_xml,
        run_id=args.run_id,
        parent_run=args.parent_run,
        input_text=input_text,
    )

    ledger = None
    if args.ledger:
        l_path = Path(args.ledger)
        if l_path.exists():
            ledger = WorkflowLedger.load_json(l_path)
        else:
            ledger = WorkflowLedger(workflow_id=args.workflow_id or "workflow_001")

    decision = plan_next_calculation(
        result=result,
        prev_input=pw_in,
        policy_name=args.policy,
        output_dir=args.output_dir,
        ledger=ledger,
        next_run_id=args.next_run_id,
    )

    if ledger and args.ledger:
        ledger.save_json(args.ledger)

    sys.stdout.write(f"Decision : {decision.decision_type}\n")
    sys.stdout.write(f"Policy   : {decision.policy_name} (v{decision.policy_version})\n")
    sys.stdout.write(f"Target   : {decision.target_calculation}\n")
    sys.stdout.write(f"Restart  : {decision.is_restart}\n")
    sys.stdout.write(f"Reason   : {decision.reason}\n")
    if args.output_dir:
        sys.stdout.write(f"Staged   : {args.output_dir}/pw.in\n")
    elif decision.next_input_text:
        sys.stdout.write("\n--- Generated pw.in ---\n")
        sys.stdout.write(decision.next_input_text)

    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Handle `qeanalyzer history` subcommand."""
    l_path = Path(args.ledger)
    if not l_path.exists():
        sys.stderr.write(f"Error: Ledger file not found at '{args.ledger}'\n")
        return 1

    ledger = WorkflowLedger.load_json(l_path)
    sys.stdout.write(f"Workflow ID : {ledger.workflow_id}\n")
    sys.stdout.write(f"Total Runs  : {len(ledger.runs)}\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.write(f"{'Run ID':<15} {'Calc':<10} {'Parent':<15} {'Status':<12} {'Policy':<15}\n")
    sys.stdout.write("-" * 70 + "\n")
    for r in ledger.runs:
        parent = r.parent_run or "-"
        policy = r.policy_applied or "-"
        sys.stdout.write(f"{r.run_id:<15} {r.calculation:<10} {parent:<15} {r.status:<12} {policy:<15}\n")

    if getattr(args, "plot", None):
        try:
            plot_path = args.plot
            plot_workflow_history(ledger, output_path=plot_path, dpi=getattr(args, "dpi", 300))
            sys.stdout.write(f"Saved workflow history plot to {plot_path}\n")
        except Exception as exc:
            sys.stderr.write(f"Error generating workflow history plot: {exc}\n")
            return 1

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Handle `qeanalyzer validate` subcommand."""
    l_path = Path(args.ledger)
    if not l_path.exists():
        sys.stderr.write(f"Error: Ledger file not found at '{args.ledger}'\n")
        return 1

    ledger = WorkflowLedger.load_json(l_path)
    errors = ledger.validate()
    if errors:
        sys.stderr.write(f"Validation FAILED ({len(errors)} errors found):\n")
        for err in errors:
            sys.stderr.write(f"  - {err}\n")
        return 1

    sys.stdout.write(f"Workflow ledger '{args.ledger}' is VALID ({len(ledger.runs)} runs, clean DAG).\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build root CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="qeanalyzer",
        description=(
            "Analyze Quantum ESPRESSO calculations and orchestrate "
            "deterministic serial QE / DFT-ADAPT workflows."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # dump subcommand
    dump_parser = subparsers.add_parser(
        "dump",
        help="Parse calculation files and dump structured canonical JSON record",
    )
    dump_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to calculation files (pw.in, pw.out, data-file-schema.xml) or directories",
    )
    dump_parser.add_argument(
        "-o", "--output",
        help="Output destination path for the generated result.json",
    )
    dump_parser.add_argument(
        "--run-id",
        help="Optional unique calculation run identifier",
    )
    dump_parser.add_argument(
        "--parent-run",
        help="Optional parent run identifier in the workflow DAG",
    )
    dump_parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation spaces for JSON output (default: 2)",
    )

    # report subcommand
    report_parser = subparsers.add_parser(
        "report",
        help="Generate human-readable text or Markdown analysis report",
    )
    report_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to calculation files (pw.in, pw.out, data-file-schema.xml) or directories",
    )
    report_parser.add_argument(
        "-o", "--output",
        help="Output destination path for the generated report",
    )
    report_parser.add_argument(
        "--markdown",
        action="store_true",
        help="Generate report in GitHub-flavored Markdown format",
    )
    report_parser.add_argument(
        "--run-id",
        help="Optional unique calculation run identifier",
    )
    report_parser.add_argument(
        "--parent-run",
        help="Optional parent run identifier in the workflow DAG",
    )

    # plot subcommand
    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate convergence visualization figures",
    )
    plot_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to calculation files (pw.out, data-file-schema.xml, workflow.json) or directories",
    )
    plot_parser.add_argument(
        "-o", "--output",
        help="Output destination path for the plot figure (e.g. scf_conv.png, workflow_history.png)",
    )
    plot_parser.add_argument(
        "--what",
        choices=["scf", "relax", "history"],
        help="Type of convergence plot to generate ('scf', 'relax', or 'history')",
    )
    plot_parser.add_argument(
        "--title",
        help="Custom figure title",
    )
    plot_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution DPI (default: 300)",
    )
    plot_parser.add_argument(
        "--run-id",
        help="Optional unique calculation run identifier",
    )
    plot_parser.add_argument(
        "--parent-run",
        help="Optional parent run identifier in the workflow DAG",
    )

    # plot-history subcommand
    plot_hist_parser = subparsers.add_parser(
        "plot-history",
        help="Generate workflow lineage and multi-run history visualization plot",
    )
    plot_hist_parser.add_argument(
        "ledger",
        nargs="?",
        default="workflow.json",
        help="Path to workflow.json ledger file (default: 'workflow.json')",
    )
    plot_hist_parser.add_argument(
        "-o", "--output",
        default="workflow_history.png",
        help="Output destination path for the history plot (default: 'workflow_history.png')",
    )
    plot_hist_parser.add_argument(
        "--title",
        help="Custom figure title",
    )
    plot_hist_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution DPI (default: 300)",
    )

    # next subcommand
    next_parser = subparsers.add_parser(
        "next",
        help="Generate deterministic next-step input or apply automated recovery",
    )
    next_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to current calculation files (pw.in, pw.out, data-file-schema.xml) or directory",
    )
    next_parser.add_argument(
        "-o", "--output-dir",
        help="Destination directory to stage the next calculation",
    )
    next_parser.add_argument(
        "--policy",
        help="Explicit policy name to execute (e.g. 'scf_to_nscf', 'relax_to_scf')",
    )
    next_parser.add_argument(
        "--ledger",
        help="Path to workflow ledger JSON file (e.g. workflow.json)",
    )
    next_parser.add_argument(
        "--workflow-id",
        help="Workflow identifier for ledger initialization",
    )
    next_parser.add_argument(
        "--run-id",
        help="Current calculation run identifier",
    )
    next_parser.add_argument(
        "--next-run-id",
        help="Identifier for the new derived calculation run",
    )
    next_parser.add_argument(
        "--parent-run",
        help="Parent run identifier",
    )

    # history subcommand
    hist_parser = subparsers.add_parser(
        "history",
        help="Display workflow execution history from ledger",
    )
    hist_parser.add_argument(
        "ledger",
        nargs="?",
        default="workflow.json",
        help="Path to workflow.json ledger file (default: 'workflow.json')",
    )
    hist_parser.add_argument(
        "--plot",
        help="Optional path to save workflow history visualization plot",
    )
    hist_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution DPI for plot (default: 300)",
    )

    # validate subcommand
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate workflow ledger integrity and DAG lineage",
    )
    val_parser.add_argument(
        "ledger",
        nargs="?",
        default="workflow.json",
        help="Path to workflow.json ledger file (default: 'workflow.json')",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dump":
        return cmd_dump(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "plot":
        return cmd_plot(args)
    if args.command == "plot-history":
        return cmd_plot_history(args)
    if args.command == "next":
        return cmd_next(args)
    if args.command == "history":
        return cmd_history(args)
    if args.command == "validate":
        return cmd_validate(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
