"""Base abstractions and specifications for calculation runners."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

JobState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"]


@dataclass
class JobStatus:
    """Execution status and metadata of a submitted calculation job."""

    job_id: str
    state: JobState
    exit_code: int | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    error_message: str = ""
    timing_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert job status to dictionary."""
        return {
            "job_id": self.job_id,
            "state": self.state,
            "exit_code": self.exit_code,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "error_message": self.error_message,
            "timing_seconds": self.timing_seconds,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobStatus:
        """Construct JobStatus from dictionary."""
        return cls(
            job_id=data["job_id"],
            state=data.get("state", "UNKNOWN"),
            exit_code=data.get("exit_code"),
            stdout_path=data.get("stdout_path"),
            stderr_path=data.get("stderr_path"),
            error_message=data.get("error_message", ""),
            timing_seconds=data.get("timing_seconds"),
            metadata=data.get("metadata", {}),
        )

    def is_finished(self) -> bool:
        """Check if job has finished execution (completed, failed, or cancelled)."""
        return self.state in ("COMPLETED", "FAILED", "CANCELLED")


@dataclass
class RunSpec:
    """Specification of an executable calculation run."""

    working_dir: str | Path
    command: list[str] = field(default_factory=lambda: ["pw.x", "-in", "pw.in"])
    input_file: str | Path | None = "pw.in"
    output_file: str | Path | None = "pw.out"
    error_file: str | Path | None = "pw.err"
    env: dict[str, str] | None = None
    n_procs: int = 1
    n_threads: int = 1
    timeout_seconds: float | None = None
    job_name: str = "qe_calc"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert run spec to dictionary."""
        return {
            "working_dir": str(self.working_dir),
            "command": list(self.command),
            "input_file": str(self.input_file) if self.input_file else None,
            "output_file": str(self.output_file) if self.output_file else None,
            "error_file": str(self.error_file) if self.error_file else None,
            "env": dict(self.env) if self.env else None,
            "n_procs": self.n_procs,
            "n_threads": self.n_threads,
            "timeout_seconds": self.timeout_seconds,
            "job_name": self.job_name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSpec:
        """Construct RunSpec from dictionary."""
        return cls(
            working_dir=data["working_dir"],
            command=data.get("command", ["pw.x", "-in", "pw.in"]),
            input_file=data.get("input_file"),
            output_file=data.get("output_file"),
            error_file=data.get("error_file"),
            env=data.get("env"),
            n_procs=data.get("n_procs", 1),
            n_threads=data.get("n_threads", 1),
            timeout_seconds=data.get("timeout_seconds"),
            job_name=data.get("job_name", "qe_calc"),
            metadata=data.get("metadata", {}),
        )


class BaseRunner(ABC):
    """Abstract base class for calculation execution environments."""

    @abstractmethod
    def submit(self, run_spec: RunSpec) -> JobStatus:
        """Submit or launch a calculation run."""
        ...

    @abstractmethod
    def status(self, job_id: str, working_dir: str | Path | None = None) -> JobStatus:
        """Query the status of a calculation job."""
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel or terminate a calculation job."""
        ...

    def wait(
        self,
        job_id: str,
        working_dir: str | Path | None = None,
        poll_interval_seconds: float = 0.5,
        timeout_seconds: float | None = None,
        max_unknown_polls: int = 10,
    ) -> JobStatus:
        """Poll and wait until job execution is finished or timeout expires.

        An UNKNOWN state is never terminal, but it is also not progress: a job the
        backend cannot identify will never reach a finished state, so repeated
        UNKNOWN polls would otherwise spin forever when *timeout_seconds* is None.
        After *max_unknown_polls* consecutive UNKNOWN results the wait gives up and
        reports FAILED. A single transient UNKNOWN does not end the wait, since the
        counter resets as soon as any other state is observed.
        """
        start_time = time.time()
        unknown_polls = 0
        while True:
            st = self.status(job_id, working_dir=working_dir)
            if st.is_finished():
                return st

            if st.state == "UNKNOWN":
                unknown_polls += 1
                if unknown_polls >= max_unknown_polls:
                    return JobStatus(
                        job_id=job_id,
                        state="FAILED",
                        error_message=(
                            f"Job state was UNKNOWN on {unknown_polls} consecutive polls; "
                            "the backend cannot identify this job. "
                            + (st.error_message or "")
                        ).strip(),
                    )
            else:
                unknown_polls = 0

            if timeout_seconds is not None and (time.time() - start_time) > timeout_seconds:
                self.cancel(job_id)
                return JobStatus(
                    job_id=job_id,
                    state="FAILED",
                    error_message=f"Timed out after {timeout_seconds} seconds.",
                )

            time.sleep(poll_interval_seconds)
