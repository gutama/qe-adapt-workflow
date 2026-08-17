"""Tests for runner abstraction layer (src/qeanalyzer/runner/)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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

            # Check status
            st = runner.status(status.job_id)
            self.assertEqual(st.state, "QUEUED")

            # Cancel
            cancelled = runner.cancel(status.job_id)
            self.assertTrue(cancelled)
            st_after = runner.status(status.job_id)
            self.assertEqual(st_after.state, "CANCELLED")


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


if __name__ == "__main__":
    unittest.main()
