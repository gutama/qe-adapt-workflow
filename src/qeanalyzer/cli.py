"""Command-line entry point for qeanalyzer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .io import PWInput, PWOutput, QEXMLOutput, read_pw_input, read_pw_output, read_qe_xml
from .models import build_run_result
from .plotting import plot_relaxation_convergence, plot_scf_convergence
from .report import dump_result_json, generate_text_report, save_result_json, save_text_report


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
            sys.stderr.write(f"Error: Unknown plot target '{what}'. Choose 'scf' or 'relax'.\n")
            return 1
    except Exception as exc:
        sys.stderr.write(f"Error generating plot: {exc}\n")
        return 1

    sys.stdout.write(f"Saved {what} convergence plot to {output_path}\n")
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
        help="Paths to calculation files (pw.out, data-file-schema.xml) or directories",
    )
    plot_parser.add_argument(
        "-o", "--output",
        help="Output destination path for the plot figure (e.g. scf_conv.png)",
    )
    plot_parser.add_argument(
        "--what",
        choices=["scf", "relax"],
        help="Type of convergence plot to generate ('scf' or 'relax')",
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
