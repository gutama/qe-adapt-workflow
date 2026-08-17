"""Workflow ledger and lineage tracking across calculation DAGs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class WorkflowRunEntry:
    """A single record entry in the workflow ledger."""

    run_id: str
    calculation: str
    parent_run: str | None = None
    status: str = "unknown"  # "converged", "not_converged", "interrupted", "running", "planned"
    run_dir: str | None = None
    input_file: str | None = None
    output_file: str | None = None
    xml_file: str | None = None
    result_json: str | None = None
    timestamp: str | None = None
    energy_ry: float | None = None
    policy_applied: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "run_id": self.run_id,
            "calculation": self.calculation,
            "parent_run": self.parent_run,
            "status": self.status,
            "run_dir": self.run_dir,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "xml_file": self.xml_file,
            "result_json": self.result_json,
            "timestamp": self.timestamp,
            "energy_ry": self.energy_ry,
            "policy_applied": self.policy_applied,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowRunEntry:
        """Construct entry from dictionary."""
        return cls(
            run_id=data["run_id"],
            calculation=data.get("calculation", "scf"),
            parent_run=data.get("parent_run"),
            status=data.get("status", "unknown"),
            run_dir=data.get("run_dir"),
            input_file=data.get("input_file"),
            output_file=data.get("output_file"),
            xml_file=data.get("xml_file"),
            result_json=data.get("result_json"),
            timestamp=data.get("timestamp"),
            energy_ry=data.get("energy_ry"),
            policy_applied=data.get("policy_applied"),
        )


@dataclass
class WorkflowLedger:
    """Versioned workflow ledger recording calculation sequences and lineage."""

    schema_version: str = "1.0"
    workflow_id: str = "default_workflow"
    description: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    runs: list[WorkflowRunEntry] = field(default_factory=list)

    def add_run(self, entry: WorkflowRunEntry) -> None:
        """Append or update a run entry in the ledger."""
        for i, r in enumerate(self.runs):
            if r.run_id == entry.run_id:
                self.runs[i] = entry
                self.updated_at = datetime.now(timezone.utc).isoformat()
                return
        self.runs.append(entry)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_run(self, run_id: str) -> WorkflowRunEntry | None:
        """Lookup run entry by ID."""
        for r in self.runs:
            if r.run_id == run_id:
                return r
        return None

    def get_latest_run(self) -> WorkflowRunEntry | None:
        """Return the most recently recorded run."""
        if not self.runs:
            return None
        return self.runs[-1]

    def get_lineage(self, run_id: str) -> list[WorkflowRunEntry]:
        """Traverse the DAG upwards to retrieve the complete ancestor lineage of a run."""
        lineage: list[WorkflowRunEntry] = []
        curr_id = run_id
        visited: set[str] = set()

        while curr_id and curr_id not in visited:
            visited.add(curr_id)
            entry = self.get_run(curr_id)
            if not entry:
                break
            lineage.append(entry)
            curr_id = entry.parent_run or ""

        return list(reversed(lineage))

    def validate(self) -> list[str]:
        """Validate ledger integrity (uniqueness, parent references, cycle detection)."""
        errors: list[str] = []
        seen_ids: set[str] = set()

        for r in self.runs:
            if r.run_id in seen_ids:
                errors.append(f"Duplicate run_id detected: '{r.run_id}'")
            seen_ids.add(r.run_id)

            if r.parent_run and r.parent_run not in seen_ids and r.parent_run != "root":
                errors.append(f"Run '{r.run_id}' references unknown or forward parent_run: '{r.parent_run}'")

        # Cycle check
        for r in self.runs:
            curr = r.parent_run
            path = {r.run_id}
            while curr:
                if curr in path:
                    errors.append(f"Cyclic dependency detected involving run '{curr}'")
                    break
                path.add(curr)
                parent_entry = self.get_run(curr)
                curr = parent_entry.parent_run if parent_entry else None

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize ledger to dictionary."""
        return {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "runs": [r.to_dict() for r in self.runs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowLedger:
        """Reconstruct ledger from dictionary."""
        runs = [WorkflowRunEntry.from_dict(d) for d in data.get("runs", [])]
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            workflow_id=data.get("workflow_id", "default_workflow"),
            description=data.get("description"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            runs=runs,
        )

    def save_json(self, path: str | Path) -> None:
        """Save ledger to a JSON file."""
        data = self.to_dict()
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> WorkflowLedger:
        """Load ledger from a JSON file."""
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
        return cls.from_dict(data)
