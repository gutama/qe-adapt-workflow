"""Coherent QE run-source discovery.

Serial workflows often keep several calculations below one parent directory.
This module deliberately refuses recursive "first file wins" discovery: a run
bundle must resolve to at most one input, one text output and one QE XML record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qeanalyzer.io.pw_input import PWInput, read_pw_input
from qeanalyzer.io.pw_output import PWOutput, read_pw_output
from qeanalyzer.io.qe_xml import QEXMLOutput, read_qe_xml


@dataclass(frozen=True)
class QESourcePaths:
    input_path: Path | None = None
    output_path: Path | None = None
    xml_path: Path | None = None


def _kind(path: Path) -> str | None:
    name = path.name.lower()
    if name == "data-file-schema.xml" or path.suffix.lower() == ".xml":
        return "xml"
    if path.suffix.lower() in {".out", ".log", ".pwo"}:
        return "output"
    if path.suffix.lower() in {".in", ".pwi"}:
        return "input"
    return None


def _directory_candidates(directory: Path) -> list[Path]:
    """Scan only one run directory plus its immediate ``*.save`` XML location."""
    files = [p for p in directory.iterdir() if p.is_file() and _kind(p)]
    for savedir in sorted(directory.glob("*.save")):
        if savedir.is_dir():
            xml = savedir / "data-file-schema.xml"
            if xml.is_file():
                files.append(xml)
    # Some workflows pass prefix.save itself.
    if directory.name.endswith(".save"):
        xml = directory / "data-file-schema.xml"
        if xml.is_file():
            files.append(xml)
    return files


def resolve_qe_source_paths(paths: list[str] | tuple[str, ...]) -> QESourcePaths:
    if not paths:
        raise ValueError("at least one QE file or run directory is required")
    candidates: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"QE source path does not exist: {path}")
        if path.is_dir():
            candidates.extend(_directory_candidates(path))
        elif path.is_file():
            if _kind(path):
                candidates.append(path)

    unique = []
    seen: set[Path] = set()
    for item in candidates:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(item)

    grouped: dict[str, list[Path]] = {"input": [], "output": [], "xml": []}
    for path in unique:
        kind = _kind(path)
        if kind:
            grouped[kind].append(path)
    for kind, values in grouped.items():
        if len(values) > 1:
            rendered = ", ".join(str(v) for v in values)
            raise ValueError(
                f"Ambiguous QE run: found multiple {kind} files ({rendered}). "
                "Pass one run directory or explicit matching files instead of a workflow parent."
            )
    if not any(grouped.values()):
        raise ValueError("No QE input/output/XML source found")
    return QESourcePaths(
        input_path=grouped["input"][0] if grouped["input"] else None,
        output_path=grouped["output"][0] if grouped["output"] else None,
        xml_path=grouped["xml"][0] if grouped["xml"] else None,
    )


def detect_and_load_sources(
    paths: list[str] | tuple[str, ...],
) -> tuple[PWInput | None, PWOutput | None, QEXMLOutput | None, str | None]:
    resolved = resolve_qe_source_paths(paths)
    pw_in = pw_out = qe_xml = None
    input_text = None
    if resolved.input_path:
        try:
            input_text = resolved.input_path.read_text(encoding="utf-8")
            pw_in = read_pw_input(resolved.input_path)
        except Exception as exc:
            raise ValueError(f"Failed to parse QE input {resolved.input_path}: {exc}") from exc
    if resolved.output_path:
        try:
            pw_out = read_pw_output(resolved.output_path)
        except Exception as exc:
            raise ValueError(f"Failed to parse QE output {resolved.output_path}: {exc}") from exc
    if resolved.xml_path:
        try:
            qe_xml = read_qe_xml(resolved.xml_path)
        except Exception as exc:
            raise ValueError(f"Failed to parse QE XML {resolved.xml_path}: {exc}") from exc
    return pw_in, pw_out, qe_xml, input_text
