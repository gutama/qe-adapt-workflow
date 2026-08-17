"""Local process calculation runner using subprocess."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from qeanalyzer.runner.base import BaseRunner, JobStatus, RunSpec


class LocalRunner(BaseRunner):
    """Executes calculation runs locally using subprocesses.

    Parameters
    ----------
    mpirun_cmd : str, optional
        MPI executable to prefix parallel runs (e.g. 'mpirun' or 'mpiexec').
    async_mode : bool, optional
        If True, submit() returns immediately with RUNNING status (default: False).
    """

    def __init__(self, mpirun_cmd: str | None = None, async_mode: bool = False) -> None:
        self.mpirun_cmd = mpirun_cmd
        self.async_mode = async_mode
        self._active_processes: dict[str, subprocess.Popen] = {}
        self._open_files: dict[str, tuple[Any, Any]] = {}
        self._start_times: dict[str, float] = {}

    def submit(self, run_spec: RunSpec) -> JobStatus:
        """Launch local process in the specified working directory."""
        work_dir = Path(run_spec.working_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Build full command
        cmd = list(run_spec.command)
        if run_spec.n_procs > 1 and self.mpirun_cmd:
            cmd = [self.mpirun_cmd, "-np", str(run_spec.n_procs)] + cmd

        # Configure environment
        env = dict(os.environ)
        if run_spec.env:
            env.update(run_spec.env)
        if run_spec.n_threads > 1:
            env["OMP_NUM_THREADS"] = str(run_spec.n_threads)

        out_path = work_dir / (run_spec.output_file or "pw.out")
        err_path = work_dir / (run_spec.error_file or "pw.err")

        stdout_file = open(out_path, "wb")
        stderr_file = open(err_path, "wb")

        start_time = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(work_dir),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
            )
        except Exception as exc:
            stdout_file.close()
            stderr_file.close()
            return JobStatus(
                job_id="failed_launch",
                state="FAILED",
                exit_code=-1,
                stdout_path=str(out_path),
                stderr_path=str(err_path),
                error_message=str(exc),
            )

        job_id = f"local_{proc.pid}"
        self._start_times[job_id] = start_time

        if self.async_mode:
            self._active_processes[job_id] = proc
            self._open_files[job_id] = (stdout_file, stderr_file)
            return JobStatus(
                job_id=job_id,
                state="RUNNING",
                stdout_path=str(out_path),
                stderr_path=str(err_path),
                metadata={"pid": proc.pid, "command": cmd},
            )

        # Synchronous execution
        try:
            proc.wait(timeout=run_spec.timeout_seconds)
            exit_code = proc.returncode
            state = "COMPLETED" if exit_code == 0 else "FAILED"
            timing = time.time() - start_time
        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = -1
            state = "FAILED"
            timing = time.time() - start_time
        finally:
            stdout_file.close()
            stderr_file.close()

        return JobStatus(
            job_id=job_id,
            state=state,
            exit_code=exit_code,
            stdout_path=str(out_path),
            stderr_path=str(err_path),
            timing_seconds=round(timing, 3),
            metadata={"pid": proc.pid, "command": cmd},
        )

    def status(self, job_id: str, working_dir: str | Path | None = None) -> JobStatus:
        """Check status of active background process."""
        proc = self._active_processes.get(job_id)
        if proc is None:
            return JobStatus(job_id=job_id, state="UNKNOWN", error_message="Job not found in local registry.")

        poll = proc.poll()
        if poll is None:
            return JobStatus(job_id=job_id, state="RUNNING")

        start = self._start_times.get(job_id, time.time())
        timing = time.time() - start
        state = "COMPLETED" if poll == 0 else "FAILED"

        # Cleanup process and files from registry
        del self._active_processes[job_id]
        if job_id in self._open_files:
            f_out, f_err = self._open_files.pop(job_id)
            try:
                f_out.close()
                f_err.close()
            except Exception:
                pass

        return JobStatus(
            job_id=job_id,
            state=state,
            exit_code=poll,
            timing_seconds=round(timing, 3),
        )

    def cancel(self, job_id: str) -> bool:
        """Terminate active process."""
        proc = self._active_processes.get(job_id)
        if proc is None:
            return False

        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            proc.kill()

        if job_id in self._active_processes:
            del self._active_processes[job_id]
        if job_id in self._open_files:
            f_out, f_err = self._open_files.pop(job_id)
            try:
                f_out.close()
                f_err.close()
            except Exception:
                pass
        return True
