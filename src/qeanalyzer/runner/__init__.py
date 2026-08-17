"""Execution runner abstractions for local, cluster, and mock execution."""

from __future__ import annotations

from typing import Any

from qeanalyzer.runner.base import BaseRunner, JobState, JobStatus, RunSpec
from qeanalyzer.runner.local import LocalRunner
from qeanalyzer.runner.mock import MockRunner
from qeanalyzer.runner.slurm import SlurmRunner


def create_runner(runner_type: str = "local", **kwargs: Any) -> BaseRunner:
    """Factory to create a runner instance.

    Parameters
    ----------
    runner_type : str, optional
        'local' (default), 'slurm', or 'mock'.
    """
    rt = runner_type.lower()
    if rt == "local":
        return LocalRunner(**kwargs)
    if rt == "slurm":
        return SlurmRunner(**kwargs)
    if rt == "mock":
        return MockRunner(**kwargs)
    raise ValueError(f"Unknown runner type '{runner_type}'. Choose from 'local', 'slurm', or 'mock'.")


__all__ = [
    "BaseRunner",
    "JobState",
    "JobStatus",
    "LocalRunner",
    "MockRunner",
    "RunSpec",
    "SlurmRunner",
    "create_runner",
]
