"""Recovery policies: the single variable of the experiment.

Owner: 丙 (policies & experiments)
Contract (frozen, see docs/interface.md):
  - Policies only DECIDE; they never call the model, never touch the backend,
    and never read ground truth.
  - on_failure returns one RecoveryAction; the agent loop carries it out and
    counts budgets, so all arms share the identical budget machinery.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FailureInfo:
    """Everything a policy may use to decide. Read-only."""
    logical_call: int          # logical tool call index (retries keep it)
    error_type: str            # "Timeout" | "RateLimit" | "Validation"
    message: str
    transient: bool            # True for Timeout/RateLimit
    attempts_on_call: int      # executions of this logical call so far (incl. this failure)
    repair_turns_used: int     # model re-plans already spent this run
    attempts_budget_left: int  # tool-execution budget remaining this run


@dataclass
class RecoveryAction:
    kind: str                  # "retry" | "replan" | "give_up"
    max_retries: int = 0       # extra attempts allowed before re-deciding
    backoff_seconds: float = 0.0


class RecoveryPolicy(ABC):
    name = "base"

    @abstractmethod
    def on_failure(self, info: FailureInfo) -> RecoveryAction:
        """Decide what to do after a failed tool execution."""


# C0: plain retry ------------------------------------------------

class RetryOnlyPolicy(RecoveryPolicy):
    """Blind retry: re-issue the identical call, never consult the model.

    The run uses a single model call (runner sets max_model_calls=1 for c0),
    so tasks needing any re-planning (semantic errors, chained tools) fail.
    """

    name = "c0_retry_only"

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def on_failure(self, info):
        if info.attempts_on_call <= self.max_retries and info.attempts_budget_left > 0:
            return RecoveryAction("retry")
        return RecoveryAction("give_up")


# C1: typed error feedback ----------------------------------------

class FeedbackPolicy(RecoveryPolicy):
    """On any failure, append (error_type, message) to the conversation and
    let the model re-plan freely."""

    name = "c1_typed_feedback"

    def on_failure(self, info):
        if info.attempts_budget_left > 0:
            return RecoveryAction("replan")
        return RecoveryAction("give_up")


# C2: bounded repair ----------------------------------------------

class BoundedRepairPolicy(RecoveryPolicy):
    """Treat transient and semantic errors differently, within fixed budgets:
      - transient (Timeout/RateLimit): retry with backoff, no model consult;
      - semantic (Validation): one model re-plan with the typed error;
      - repair turns (re-plans) are capped; beyond the cap, give up.
    """

    name = "c2_bounded_repair"

    def __init__(self, max_transient_retries: int = 2, max_repair_turns: int = 2,
                 backoff_seconds: float = 1.0):
        self.max_transient_retries = max_transient_retries
        self.max_repair_turns = max_repair_turns
        self.backoff_seconds = backoff_seconds

    def on_failure(self, info):
        if info.attempts_budget_left <= 0:
            return RecoveryAction("give_up")
        if info.transient and info.attempts_on_call <= self.max_transient_retries:
            return RecoveryAction("retry", backoff_seconds=self.backoff_seconds)
        if info.repair_turns_used < self.max_repair_turns:
            return RecoveryAction("replan")
        return RecoveryAction("give_up")


POLICY_REGISTRY = {
    RetryOnlyPolicy.name: RetryOnlyPolicy,
    FeedbackPolicy.name: FeedbackPolicy,
    BoundedRepairPolicy.name: BoundedRepairPolicy,
}


def make_policy(name: str) -> RecoveryPolicy:
    try:
        return POLICY_REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown policy {name!r}; available: {sorted(POLICY_REGISTRY)}")
