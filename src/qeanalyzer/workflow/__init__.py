"""Workflow state machine, policy engine, and recovery layer."""

from qeanalyzer.workflow.decisions import DecisionType, NextRunDecision
from qeanalyzer.workflow.ledger import WorkflowLedger, WorkflowRunEntry
from qeanalyzer.workflow.orchestration import plan_next_calculation
from qeanalyzer.workflow.outer_loop import (
    ConvergenceCriteria,
    ConvergenceCheckResult,
    OuterLoopController,
    OuterLoopIterationRecord,
    OuterLoopLedger,
)
from qeanalyzer.workflow.policy import Policy, PolicyRegistry, default_registry
from qeanalyzer.workflow.rules import (
    InterruptedRecoveryPolicy,
    RelaxToSCFPolicy,
    SCFToNSCFPolicy,
    UnconvergedSCFPolicy,
)

# Register built-in policies into the global default registry
default_registry.register(InterruptedRecoveryPolicy())
default_registry.register(UnconvergedSCFPolicy())
default_registry.register(RelaxToSCFPolicy())
default_registry.register(SCFToNSCFPolicy())

__all__ = [
    "ConvergenceCheckResult",
    "ConvergenceCriteria",
    "DecisionType",
    "InterruptedRecoveryPolicy",
    "NextRunDecision",
    "OuterLoopController",
    "OuterLoopIterationRecord",
    "OuterLoopLedger",
    "Policy",
    "PolicyRegistry",
    "RelaxToSCFPolicy",
    "SCFToNSCFPolicy",
    "UnconvergedSCFPolicy",
    "WorkflowLedger",
    "WorkflowRunEntry",
    "default_registry",
    "plan_next_calculation",
]
