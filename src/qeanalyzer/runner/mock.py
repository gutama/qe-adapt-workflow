"""Mock calculation runner for deterministic testing and synthetic workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qeanalyzer.runner.base import BaseRunner, JobStatus, RunSpec


class MockRunner(BaseRunner):
    """Deterministic in-memory mock runner for integration and policy testing.

    Parameters
    ----------
    default_state : str, optional
        JobState to return on submit (default: 'COMPLETED').
    default_exit_code : int, optional
        Returncode to report (default: 0).
    stdout_content : str, optional
        Synthetic stdout content to write to output_file if present.
    """

    def __init__(
        self,
        default_state: str = "COMPLETED",
        default_exit_code: int = 0,
        stdout_content: str | None = None,
    ) -> None:
        self.default_state = default_state
        self.default_exit_code = default_exit_code
        self.stdout_content = stdout_content
        self.submitted_specs: list[RunSpec] = []
        self._jobs: dict[str, JobStatus] = {}

    def submit(self, run_spec: RunSpec) -> JobStatus:
        """Record submission and optionally write mock stdout content."""
        self.submitted_specs.append(run_spec)
        job_id = f"mock_{len(self.submitted_specs)}"

        work_dir = Path(run_spec.working_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        out_path = work_dir / (run_spec.output_file or "pw.out")
        err_path = work_dir / (run_spec.error_file or "pw.err")

        if self.stdout_content is not None:
            out_path.write_text(self.stdout_content, encoding="utf-8")

        status = JobStatus(
            job_id=job_id,
            state=self.default_state,  # type: ignore
            exit_code=self.default_exit_code,
            stdout_path=str(out_path),
            stderr_path=str(err_path),
            timing_seconds=0.05,
            metadata={"command": run_spec.command},
        )
        self._jobs[job_id] = status
        return status

    def status(self, job_id: str, working_dir: str | Path | None = None) -> JobStatus:
        """Retrieve recorded job status."""
        return self._jobs.get(
            job_id,
            JobStatus(job_id=job_id, state="UNKNOWN", error_message="Job not found in MockRunner."),
        )

    def cancel(self, job_id: str) -> bool:
        """Cancel mock job."""
        if job_id in self._jobs:
            self._jobs[job_id].state = "CANCELLED"
            return True
        return False
