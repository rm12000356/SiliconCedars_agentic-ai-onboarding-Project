from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_CALLS = 20
DEFAULT_MAX_TOKENS = 60000
DEFAULT_MAX_SECONDS = 120.0


class TurnBudgetExceeded(RuntimeError):
    """Raised when a single user turn exceeds its LLM call/token/time budget."""

    def __init__(self, kind: str, limit: float, observed: float):
        self.kind = kind
        self.limit = limit
        self.observed = observed
        super().__init__(f"Turn {kind} budget exceeded: {observed} >= {limit}")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


@dataclass
class TurnBudget:
    """Aggregate per-turn ceiling across all specialists and subgraphs.

    Individual nodes already cap their own loops; this is the global backstop
    that a confused turn cannot multiply past.

    The time limit measures accumulated *LLM call time*, not wall-clock since
    the turn began. Human think-time (e.g. answering a clarification) and
    tool/DB latency therefore do not count against it.
    """

    max_calls: int = DEFAULT_MAX_CALLS
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_seconds: float = DEFAULT_MAX_SECONDS
    calls: int = 0
    tokens: int = 0
    llm_seconds: float = 0.0

    @classmethod
    def from_env(cls) -> "TurnBudget":
        return cls(
            max_calls=_env_int("MAX_LLM_CALLS_PER_TURN", DEFAULT_MAX_CALLS),
            max_tokens=_env_int("MAX_TOKENS_PER_TURN", DEFAULT_MAX_TOKENS),
            max_seconds=_env_float("MAX_TURN_SECONDS", DEFAULT_MAX_SECONDS),
        )

    def exhausted(self) -> bool:
        """True if another LLM call would exceed any configured limit."""
        if self.calls >= self.max_calls:
            return True
        if self.tokens >= self.max_tokens:
            return True
        return self.llm_seconds >= self.max_seconds

    def check_and_count(self) -> None:
        """Reserve one LLM call or raise if the turn is over budget."""
        if self.calls + 1 > self.max_calls:
            raise TurnBudgetExceeded("llm_call", self.max_calls, self.calls + 1)
        if self.tokens >= self.max_tokens:
            raise TurnBudgetExceeded("token", self.max_tokens, self.tokens)
        if self.llm_seconds >= self.max_seconds:
            raise TurnBudgetExceeded(
                "wall_clock", self.max_seconds, self.llm_seconds
            )
        self.calls += 1

    def record_duration(self, seconds: float) -> None:
        """Accumulate the elapsed time of one provider call (success or not)."""
        if seconds > 0:
            self.llm_seconds += seconds

    def record_usage(self, response) -> None:
        """Best-effort token accumulation from a model response.

        Structured-output responses are often parsed objects without
        ``usage_metadata``; when absent only the call count applies.
        """
        usage = getattr(response, "usage_metadata", None)
        if not isinstance(usage, dict):
            return
        total = usage.get("total_tokens")
        if isinstance(total, int) and total > 0:
            self.tokens += total


_budget_var: contextvars.ContextVar[Optional[TurnBudget]] = contextvars.ContextVar(
    "turn_budget", default=None
)


def get_budget() -> Optional[TurnBudget]:
    """The active turn budget, or None when running outside any turn scope."""
    return _budget_var.get()


@contextmanager
def budget_scope(budget: Optional[TurnBudget] = None):
    """Bind a fresh budget to the current context for one user turn."""
    if budget is None:
        budget = TurnBudget.from_env()
    token = _budget_var.set(budget)
    try:
        yield budget
    finally:
        _budget_var.reset(token)
