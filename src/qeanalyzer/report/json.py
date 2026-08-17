"""JSON serialization and deserialization for canonical QERunResult records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qeanalyzer.models import QERunResult


def dump_result_json(result: QERunResult, indent: int = 2) -> str:
    """Serialize a QERunResult to a deterministic JSON formatted string.

    Parameters
    ----------
    result : QERunResult
        The calculation result to serialize.
    indent : int, optional
        Indentation spaces for pretty-printing (default: 2).

    Returns
    -------
    str
        Deterministic JSON string representation.
    """
    data = result.to_dict()
    return json.dumps(data, indent=indent, sort_keys=True) + "\n"


def save_result_json(result: QERunResult, path: str | Path, indent: int = 2) -> None:
    """Save a QERunResult to a JSON file on disk.

    Parameters
    ----------
    result : QERunResult
        The calculation result to save.
    path : str or Path
        Target destination file path.
    indent : int, optional
        Indentation spaces (default: 2).
    """
    content = dump_result_json(result, indent=indent)
    Path(path).write_text(content, encoding="utf-8")


def load_result_json(path: str | Path) -> QERunResult:
    """Load and reconstruct a QERunResult from a JSON file.

    Parameters
    ----------
    path : str or Path
        Source JSON file path.

    Returns
    -------
    QERunResult
        Reconstructed calculation result.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    return QERunResult.from_dict(data)


def validate_result_dict(data: dict[str, Any]) -> list[str]:
    """Perform lightweight schema validation against canonical result structure.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary to validate.

    Returns
    -------
    list[str]
        List of validation error descriptions (empty list if valid).
    """
    errors: list[str] = []

    required_top = [
        "schema_version",
        "calculation",
        "status",
        "provenance",
        "electronic",
        "convergence",
    ]
    for key in required_top:
        if key not in data:
            errors.append(f"Missing required top-level key: '{key}'")

    # Status check
    status = data.get("status")
    if isinstance(status, dict):
        for req in ["completed", "scf_converged", "exit_status"]:
            if req not in status:
                errors.append(f"Missing required key in status: '{req}'")
    elif "status" in data:
        errors.append("Field 'status' must be an object")

    # Provenance check
    prov = data.get("provenance")
    if isinstance(prov, dict):
        for req in ["schema_version", "engine", "qeanalyzer_version"]:
            if req not in prov:
                errors.append(f"Missing required key in provenance: '{req}'")
    elif "provenance" in data:
        errors.append("Field 'provenance' must be an object")

    return errors
