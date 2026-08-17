#!/usr/bin/env python3
"""Example 1: Parsing Quantum ESPRESSO outputs, extracting canonical models, and generating reports."""

from pathlib import Path
from qeanalyzer.io import read_pw_input, read_pw_output, read_qe_xml
from qeanalyzer.models import build_run_result
from qeanalyzer.report import dump_result_json, generate_text_report

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"

def main() -> None:
    print("=" * 60)
    print(" Example 1: QE Parser & Report Generator")
    print("=" * 60)

    # 1. Parse input, text output, and XML schema
    pw_in = read_pw_input(FIXTURES / "si_scf.in")
    pw_out = read_pw_output(FIXTURES / "si_scf.out")
    qe_xml = read_qe_xml(FIXTURES / "si_scf.xml")

    # 2. Build canonical QERunResult representation
    result = build_run_result(
        pw_in=pw_in,
        pw_out=pw_out,
        qe_xml=qe_xml,
        run_id="si_scf_demo",
    )

    # 3. Generate human-readable plain text report
    report_text = generate_text_report(result, markdown=False)
    print("\n--- Summary Report ---")
    print(report_text)

    # 4. JSON serialization
    json_str = dump_result_json(result, indent=2)
    print(f"Generated JSON Schema document ({len(json_str)} bytes)")


if __name__ == "__main__":
    main()
