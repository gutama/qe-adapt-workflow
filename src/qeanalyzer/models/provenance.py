"""Provenance and metadata data models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from qeanalyzer import __version__ as QEANALYZER_VERSION


@dataclass
class QEProvenance:
    """Provenance records tracking calculations, input hashes, and execution environment."""

    schema_version: str = "1.0"
    run_id: str | None = None
    parent_run: str | None = None
    engine_name: str = "Quantum ESPRESSO"
    executable: str = "pw.x"
    qe_version: str | None = None
    qeanalyzer_version: str = QEANALYZER_VERSION
    input_sha256: str | None = None
    pseudo_sha256: dict[str, str] = field(default_factory=dict)
    prefix: str | None = None
    outdir: str | None = None
    n_mpi_processes: int | None = None
    n_threads: int | None = None
    policy_name: str | None = None
    policy_version: str | None = None

    @classmethod
    def compute_sha256(cls, text: str) -> str:
        """Compute SHA256 hex digest of a string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance metadata to a JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "parent_run": self.parent_run,
            "engine": {
                "name": self.engine_name,
                "executable": self.executable,
                "version": self.qe_version,
            },
            "qeanalyzer_version": self.qeanalyzer_version,
            "input_sha256": self.input_sha256,
            "pseudo_sha256": dict(self.pseudo_sha256),
            "prefix": self.prefix,
            "outdir": self.outdir,
            "parallel": {
                "n_mpi_processes": self.n_mpi_processes,
                "n_threads": self.n_threads,
            },
            "policy": {
                "name": self.policy_name,
                "version": self.policy_version,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QEProvenance:
        """Construct a QEProvenance instance from a dictionary."""
        engine = data.get("engine", {})
        parallel = data.get("parallel", {})
        policy = data.get("policy", {})
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            run_id=data.get("run_id"),
            parent_run=data.get("parent_run"),
            engine_name=engine.get("name", "Quantum ESPRESSO"),
            executable=engine.get("executable", "pw.x"),
            qe_version=engine.get("version"),
            qeanalyzer_version=data.get("qeanalyzer_version", QEANALYZER_VERSION),
            input_sha256=data.get("input_sha256"),
            pseudo_sha256=data.get("pseudo_sha256", {}),
            prefix=data.get("prefix"),
            outdir=data.get("outdir"),
            n_mpi_processes=parallel.get("n_mpi_processes"),
            n_threads=parallel.get("n_threads"),
            policy_name=policy.get("name"),
            policy_version=policy.get("version"),
        )
