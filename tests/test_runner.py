"""Tests for runner abstraction layer (src/qeanalyzer/runner/)."""

from __future__ import annotations

import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from qeanalyzer.runner import (
    BaseRunner,
    JobStatus,
    LocalRunner,
    MockRunner,
    RunSpec,
    SlurmRunner,
    create_runner,
)


class TestRunSpecAndJobStatus(unittest.TestCase):
    """Test serialization and data models."""

    def test_run_spec_serialization(self):
        spec = RunSpec(
            working_dir="/tmp/test_dir",
            command=["pw.x", "-in", "pw.in"],
            n_procs=8,
            n_threads=2,
            timeout_seconds=3600.0,
            job_name="si_scf",
            env={"OMP_NUM_THREADS": "2"},
        )
        d = spec.to_dict()
        rec = RunSpec.from_dict(d)
        self.assertEqual(rec.command, ["pw.x", "-in", "pw.in"])
        self.assertEqual(rec.n_procs, 8)
        self.assertEqual(rec.job_name, "si_scf")

    def test_job_status_methods(self):
        st_running = JobStatus(job_id="123", state="RUNNING")
        self.assertFalse(st_running.is_finished())

        st_done = JobStatus(job_id="123", state="COMPLETED", exit_code=0)
        self.assertTrue(st_done.is_finished())

        d = st_done.to_dict()
        rec = JobStatus.from_dict(d)
        self.assertEqual(rec.job_id, "123")
        self.assertEqual(rec.state, "COMPLETED")
        self.assertEqual(rec.exit_code, 0)


class TestLocalRunner(unittest.TestCase):
    """Test LocalRunner synchronous, asynchronous, and cancellation behavior."""

    def test_local_sync_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = LocalRunner(async_mode=False)
            spec = RunSpec(
                working_dir=tmpdir,
                command=["python3", "-c", "print('hello from runner')"],
                output_file="out.txt",
                error_file="err.txt",
            )
            status = runner.submit(spec)
            self.assertEqual(status.state, "COMPLETED")
            self.assertEqual(status.exit_code, 0)

            out_content = (Path(tmpdir) / "out.txt").read_text()
            self.assertIn("hello from runner", out_content)

    def test_local_async_execution_and_wait(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = LocalRunner(async_mode=True)
            spec = RunSpec(
                working_dir=tmpdir,
                command=["python3", "-c", "import time; time.sleep(0.05); print('async done')"],
                output_file="async.out",
            )
            status = runner.submit(spec)
            self.assertEqual(status.state, "RUNNING")

            final_status = runner.wait(status.job_id, poll_interval_seconds=0.01, timeout_seconds=2.0)
            self.assertEqual(final_status.state, "COMPLETED")

            out_content = (Path(tmpdir) / "async.out").read_text()
            self.assertIn("async done", out_content)

    def test_local_cancel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = LocalRunner(async_mode=True)
            spec = RunSpec(
                working_dir=tmpdir,
                command=["python3", "-c", "import time; time.sleep(10.0)"],
            )
            status = runner.submit(spec)
            self.assertEqual(status.state, "RUNNING")

            cancelled = runner.cancel(status.job_id)
            self.assertTrue(cancelled)


class TestSlurmRunner(unittest.TestCase):
    """Test SLURM batch script generation and mock submission."""

    def test_generate_slurm_script(self):
        runner = SlurmRunner(
            partition="compute",
            account="mat_sci_01",
            walltime="04:00:00",
            modules=["quantum-espresso/7.2"],
            srun_cmd="srun",
        )
        spec = RunSpec(
            working_dir="/scratch/calc",
            command=["pw.x", "-in", "pw.in"],
            n_procs=16,
            n_threads=2,
            job_name="si_relax",
            output_file="pw.out",
        )
        script = runner.generate_batch_script(spec)

        self.assertIn("#!/bin/bash", script)
        self.assertIn("#SBATCH --job-name=si_relax", script)
        self.assertIn("#SBATCH --partition=compute", script)
        self.assertIn("#SBATCH --account=mat_sci_01", script)
        self.assertIn("#SBATCH --time=04:00:00", script)
        self.assertIn("#SBATCH --ntasks=16", script)
        self.assertIn("#SBATCH --cpus-per-task=2", script)
        self.assertIn("module load quantum-espresso/7.2", script)
        self.assertIn("srun pw.x -in pw.in", script)

    def test_slurm_mock_submission_and_cancel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = SlurmRunner(mock_mode=True)
            spec = RunSpec(working_dir=tmpdir, job_name="test_mock_job")
            status = runner.submit(spec)

            self.assertEqual(status.state, "QUEUED")
            self.assertTrue((Path(tmpdir) / "submit_test_mock_job.sh").exists())

            # A simulated job reaches a terminal state once polled; staying QUEUED
            # forever would make wait() spin indefinitely on a valid job id.
            st = runner.status(status.job_id)
            self.assertEqual(st.state, "COMPLETED")
            self.assertTrue(st.is_finished())

            # Cancel a freshly submitted (still queued) job
            other = runner.submit(RunSpec(working_dir=tmpdir, job_name="test_cancel_job"))
            self.assertEqual(other.state, "QUEUED")
            self.assertTrue(runner.cancel(other.job_id))
            self.assertEqual(runner.status(other.job_id).state, "CANCELLED")


class TestMockRunnerAndFactory(unittest.TestCase):
    """Test MockRunner and create_runner factory."""

    def test_mock_runner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = MockRunner(stdout_content="JOB DONE SUCCESSFULLY")
            spec = RunSpec(working_dir=tmpdir, output_file="pw.out")
            st = runner.submit(spec)

            self.assertEqual(st.state, "COMPLETED")
            self.assertEqual(len(runner.submitted_specs), 1)

            out_content = (Path(tmpdir) / "pw.out").read_text()
            self.assertEqual(out_content, "JOB DONE SUCCESSFULLY")

    def test_create_runner_factory(self):
        r_local = create_runner("local")
        self.assertIsInstance(r_local, LocalRunner)

        r_slurm = create_runner("slurm", mock_mode=True)
        self.assertIsInstance(r_slurm, SlurmRunner)

        r_mock = create_runner("mock")
        self.assertIsInstance(r_mock, MockRunner)

        with self.assertRaises(ValueError):
            create_runner("unsupported_runner_type")

class TestSlurmDoesNotFakeSubmission(unittest.TestCase):
    """Mock mode must be asked for, and an unknown job is not a successful one."""

    def test_missing_sbatch_fails_instead_of_silently_mocking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = SlurmRunner(partition="compute")
            self.assertFalse(runner.mock_mode)

            with mock.patch("qeanalyzer.runner.slurm.shutil.which", return_value=None):
                status = runner.submit(RunSpec(working_dir=tmpdir, job_name="real_job"))

            self.assertEqual(status.state, "FAILED")
            self.assertIn("sbatch", status.error_message)
            # The script is still staged so the user can submit it by hand.
            self.assertTrue((Path(tmpdir) / "submit_real_job.sh").exists())

    def test_job_unknown_to_squeue_and_sacct_is_not_reported_completed(self):
        # Pretend sbatch exists so the real squeue/sacct path is exercised rather
        # than any mock branch, then have both report nothing about the job.
        empty = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with mock.patch("qeanalyzer.runner.slurm.shutil.which", return_value="/usr/bin/sbatch"):
            runner = SlurmRunner()
            self.assertFalse(runner.mock_mode)
            with mock.patch("qeanalyzer.runner.slurm.subprocess.run", return_value=empty):
                status = runner.status("123456")

        self.assertEqual(status.state, "UNKNOWN")
        self.assertFalse(status.is_finished())


class TestWaitTerminatesOnUnknownState(unittest.TestCase):
    """wait() must not spin forever on a job the backend cannot identify."""

    def test_wait_gives_up_after_repeated_unknown(self):
        runner = LocalRunner()
        start = time.time()
        status = runner.wait(
            "no-such-job", poll_interval_seconds=0.001, max_unknown_polls=3
        )
        self.assertEqual(status.state, "FAILED")
        self.assertIn("UNKNOWN", status.error_message)
        self.assertLess(time.time() - start, 5.0)

    def test_transient_unknown_does_not_end_the_wait(self):
        """The counter resets, so one blip mid-run is tolerated."""
        states = ["UNKNOWN", "RUNNING", "UNKNOWN", "UNKNOWN", "COMPLETED"]
        seen: list[str] = []

        class _Flaky(LocalRunner):
            def status(self, job_id, working_dir=None):
                state = states[len(seen)]
                seen.append(state)
                return JobStatus(job_id=job_id, state=state)

        status = _Flaky().wait(
            "job", poll_interval_seconds=0.001, max_unknown_polls=3
        )
        self.assertEqual(status.state, "COMPLETED")
        self.assertEqual(len(seen), 5)

    def test_mock_slurm_job_can_be_waited_on(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = SlurmRunner(mock_mode=True)
            job = runner.submit(RunSpec(working_dir=tmpdir, job_name="waitable"))
            status = runner.wait(job.job_id, poll_interval_seconds=0.001)
            self.assertEqual(status.state, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
