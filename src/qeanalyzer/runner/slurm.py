"""SLURM batch scheduler submission and status runner."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from qeanalyzer.runner.base import BaseRunner, JobStatus, RunSpec


class SlurmRunner(BaseRunner):
    """Generates deterministic SLURM job scripts and manages batch submission via sbatch/squeue.

    Parameters
    ----------
    partition : str, optional
        Target SLURM partition/queue (default: None).
    account : str, optional
        SLURM allocation/account name.
    qos : str, optional
        Quality of Service flag.
    walltime : str, optional
        Wall clock limit format HH:MM:SS (default: '02:00:00').
    modules : list of str, optional
        Environment modules to load before running pw.x (e.g. ['quantum-espresso/7.2']).
    srun_cmd : str, optional
        Parallel launcher command within batch script (default: 'srun').
    mock_mode : bool, optional
        If True, simulates sbatch/squeue without executing SLURM binaries (useful for testing).
    """

    def __init__(
        self,
        partition: str | None = None,
        account: str | None = None,
        qos: str | None = None,
        walltime: str = "02:00:00",
        modules: list[str] | None = None,
        srun_cmd: str = "srun",
        mock_mode: bool = False,
    ) -> None:
        self.partition = partition
        self.account = account
        self.qos = qos
        self.walltime = walltime
        self.modules = modules or []
        self.srun_cmd = srun_cmd
        # Mock mode must be asked for explicitly. Inferring it from a missing
        # sbatch would fabricate QUEUED job ids for calculations that were never
        # submitted, and the caller would wait on jobs that do not exist.
        self.mock_mode = mock_mode
        self._mock_jobs: dict[str, JobStatus] = {}

    def generate_batch_script(self, run_spec: RunSpec) -> str:
        """Generate formatted SLURM bash script content."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={run_spec.job_name}",
            f"#SBATCH --time={self.walltime}",
            f"#SBATCH --ntasks={run_spec.n_procs}",
            f"#SBATCH --cpus-per-task={run_spec.n_threads}",
            f"#SBATCH --output={run_spec.output_file or 'pw.out'}",
            f"#SBATCH --error={run_spec.error_file or 'pw.err'}",
        ]

        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")

        lines.append("")
        lines.append("# Environment setup")
        lines.append(f"export OMP_NUM_THREADS={run_spec.n_threads}")
        for mod in self.modules:
            lines.append(f"module load {mod}")

        if run_spec.env:
            for k, v in run_spec.env.items():
                lines.append(f"export {k}={v}")

        lines.append("")
        lines.append("# Execution")
        cmd_str = " ".join(run_spec.command)
        lines.append(f"{self.srun_cmd} {cmd_str}")

        return "\n".join(lines) + "\n"

    def submit(self, run_spec: RunSpec) -> JobStatus:
        """Stage SLURM script and submit to scheduler via sbatch."""
        work_dir = Path(run_spec.working_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        script_content = self.generate_batch_script(run_spec)
        script_path = work_dir / f"submit_{run_spec.job_name}.sh"
        script_path.write_text(script_content, encoding="utf-8")

        out_path = work_dir / (run_spec.output_file or "pw.out")
        err_path = work_dir / (run_spec.error_file or "pw.err")

        if self.mock_mode:
            job_id = f"slurm_mock_{len(self._mock_jobs) + 1000}"
            status = JobStatus(
                job_id=job_id,
                state="QUEUED",
                stdout_path=str(out_path),
                stderr_path=str(err_path),
                metadata={"script_path": str(script_path), "mock": True},
            )
            self._mock_jobs[job_id] = status
            return status

        # Real sbatch invocation
        if shutil.which("sbatch") is None:
            return JobStatus(
                job_id="failed_sbatch",
                state="FAILED",
                error_message=(
                    "sbatch was not found on PATH, so nothing was submitted. The batch "
                    f"script was staged at {script_path}. Pass mock_mode=True to stage "
                    "scripts deliberately without submitting them."
                ),
                stdout_path=str(out_path),
                stderr_path=str(err_path),
                metadata={"script_path": str(script_path)},
            )

        try:
            res = subprocess.run(
                ["sbatch", str(script_path.name)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            match = re.search(r"Submitted batch job (\d+)", res.stdout)
            job_id = match.group(1) if match else "slurm_submitted"
            return JobStatus(
                job_id=job_id,
                state="QUEUED",
                stdout_path=str(out_path),
                stderr_path=str(err_path),
                metadata={"script_path": str(script_path)},
            )
        except Exception as exc:
            return JobStatus(
                job_id="failed_sbatch",
                state="FAILED",
                error_message=str(exc),
                stdout_path=str(out_path),
                stderr_path=str(err_path),
            )

    def status(self, job_id: str, working_dir: str | Path | None = None) -> JobStatus:
        """Query job state via squeue or sacct."""
        if self.mock_mode:
            mock = self._mock_jobs.get(job_id)
            if mock is None:
                return JobStatus(
                    job_id=job_id,
                    state="UNKNOWN",
                    error_message="Job not found in SLURM mock state.",
                )
            # A simulated job reaches a terminal state on its first poll. Leaving it
            # QUEUED forever would make wait() spin indefinitely on a valid job id.
            if mock.state == "QUEUED":
                mock.state = "COMPLETED"
            return mock

        try:
            res = subprocess.run(
                ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                capture_output=True,
                text=True,
            )
            sq_out = res.stdout.strip()
            if sq_out:
                state_map = {
                    "PENDING": "QUEUED",
                    "RUNNING": "RUNNING",
                    "COMPLETING": "RUNNING",
                    "COMPLETED": "COMPLETED",
                    "FAILED": "FAILED",
                    "CANCELLED": "CANCELLED",
                    "TIMEOUT": "FAILED",
                }
                return JobStatus(job_id=job_id, state=state_map.get(sq_out, "UNKNOWN"))

            # If not in squeue, check sacct
            res_acct = subprocess.run(
                ["sacct", "-j", str(job_id), "-n", "-P", "--format=State,ExitCode"],
                capture_output=True,
                text=True,
            )
            acct_lines = [ln.strip() for ln in res_acct.stdout.strip().splitlines() if ln.strip()]
            if acct_lines:
                main_state = acct_lines[0].split("|")[0]
                state_map = {
                    "COMPLETED": "COMPLETED",
                    "FAILED": "FAILED",
                    "CANCELLED": "CANCELLED",
                    "TIMEOUT": "FAILED",
                }
                return JobStatus(job_id=job_id, state=state_map.get(main_state, "UNKNOWN"))

            # Neither squeue nor sacct knows this job. That is not evidence of
            # success -- it also covers a job id that was never submitted and a
            # cluster without accounting configured.
            return JobStatus(
                job_id=job_id,
                state="UNKNOWN",
                error_message=(
                    "Job is not known to squeue or sacct; it may never have been "
                    "submitted, or job accounting may be unavailable on this cluster."
                ),
            )
        except Exception as exc:
            return JobStatus(job_id=job_id, state="UNKNOWN", error_message=str(exc))

    def cancel(self, job_id: str) -> bool:
        """Cancel batch job via scancel."""
        if self.mock_mode:
            if job_id in self._mock_jobs:
                self._mock_jobs[job_id].state = "CANCELLED"
                return True
            return False

        try:
            subprocess.run(["scancel", str(job_id)], check=True)
            return True
        except Exception:
            return False
