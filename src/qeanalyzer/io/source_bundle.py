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


SAVE_SUFFIX = ".save"
SCHEMA_NAME = "data-file-schema.xml"


def _kind(path: Path) -> str | None:
    name = path.name.lower()
    if name == SCHEMA_NAME or path.suffix.lower() == ".xml":
        return "xml"
    if path.suffix.lower() in {".out", ".log", ".pwo"}:
        return "output"
    if path.suffix.lower() in {".in", ".pwi"}:
        return "input"
    return None


def _save_directories(directory: Path) -> list[Path]:
    """``prefix.save`` directories QE wrote for a run started in ``directory``.

    QE puts them below ``outdir``, which is normally a subdirectory of the run
    directory (``outdir='./tmp'``), so immediate children are searched too. The
    scan stops at that depth: anything deeper is a workflow parent, not one run.
    """
    seen: set[Path] = set()
    savedirs: list[Path] = []
    for candidate in sorted(directory.glob(f"*{SAVE_SUFFIX}")) + sorted(
        directory.glob(f"*/*{SAVE_SUFFIX}")
    ):
        resolved = candidate.resolve()
        if candidate.is_dir() and resolved not in seen:
            seen.add(resolved)
            savedirs.append(candidate)
    return savedirs


def _xml_records(savedir: Path) -> list[Path]:
    """XML records QE wrote for one ``prefix.save`` directory.

    QE >= 6.4 writes both ``<outdir>/<prefix>.save/data-file-schema.xml`` and a
    copy at ``<outdir>/<prefix>.xml``; both are returned and collapsed later.
    """
    records = []
    schema = savedir / SCHEMA_NAME
    if schema.is_file():
        records.append(schema)
    sibling = savedir.parent / f"{savedir.name[: -len(SAVE_SUFFIX)]}.xml"
    if sibling.is_file():
        records.append(sibling)
    return records


def _directory_candidates(directory: Path) -> list[Path]:
    """Scan one run directory plus the ``outdir`` locations QE writes XML to."""
    files = [p for p in directory.iterdir() if p.is_file() and _kind(p)]
    for savedir in _save_directories(directory):
        files.extend(_xml_records(savedir))
    # Some workflows pass prefix.save itself.
    if directory.name.endswith(SAVE_SUFFIX):
        schema = directory / SCHEMA_NAME
        if schema.is_file():
            files.append(schema)
    return files


def _xml_run_key(path: Path) -> tuple[Path, str] | None:
    """The ``(outdir, prefix)`` an XML record belongs to, when it is derivable.

    ``<outdir>/<prefix>.xml`` and ``<outdir>/<prefix>.save/data-file-schema.xml``
    are one run's record written twice, so they map to the same key and never
    look like two competing runs.
    """
    if path.name.lower() == SCHEMA_NAME:
        savedir = path.parent
        if savedir.name.endswith(SAVE_SUFFIX):
            return savedir.parent.resolve(), savedir.name[: -len(SAVE_SUFFIX)]
        return None
    if path.suffix.lower() == ".xml":
        return path.parent.resolve(), path.stem
    return None


def _collapse_xml_records(paths: list[Path]) -> list[Path]:
    """Keep one path per run, preferring the canonical schema record."""
    chosen: dict[object, Path] = {}
    order: list[object] = []
    for path in paths:
        key: object = _xml_run_key(path) or path.resolve()
        if key not in chosen:
            chosen[key] = path
            order.append(key)
        elif path.name.lower() == SCHEMA_NAME:
            chosen[key] = path
    return [chosen[key] for key in order]


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
    grouped["xml"] = _collapse_xml_records(grouped["xml"])
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
