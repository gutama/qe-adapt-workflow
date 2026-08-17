"""Workflow state machine, policy engine, and recovery layer."""

from qeanalyzer.workflow.decisions import DecisionType, NextRunDecision
from qeanalyzer.workflow.policy import Policy, PolicyRegistry, default_registry
from qeanalyzer.workflow.rules import (
    InterruptedRecoveryPolicy,
    RelaxToSCFPolicy,
    SCFToNSCFPolicy,
    UnconvergedSCFPolicy,
)

# Register built-in policies into the global default registry
# Priority order: Interrupted -> Unconverged SCF -> Relax-to-SCF -> SCF-to-NSCF
default_registry.register(InterruptedRecoveryPolicy())
default_registry.register(UnconvergedSCFPolicy())
default_registry.register(RelaxToSCFPolicy())
default_registry.register(SCFToNSCFPolicy())

__all__ = [
    "DecisionType",
    "InterruptedRecoveryPolicy",
    "NextRunDecision",
    "Policy",
    "PolicyRegistry",
    "RelaxToSCFPolicy",
    "SCFToNSCFPolicy",
    "UnconvergedSCFPolicy",
    "default_registry",
]
