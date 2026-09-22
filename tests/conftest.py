from __future__ import annotations

import os
import re
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from state.state import SupervisorState
from dotenv import load_dotenv

load_dotenv()

def _has_llm_credentials() -> bool:
    return bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def _db_reachable() -> bool:
    try:
        from db.connection import get_general_connection

        with get_general_connection(acquire_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


requires_llm = pytest.mark.skipif(
    not _has_llm_credentials(),
    reason="GROQ_API_KEY or OPENROUTER_API_KEY required for LLM tests",
)

requires_db = pytest.mark.skipif(
    not _db_reachable(),
    reason="Database not reachable (check DB_* / DATABASE_URL env vars)",
)


_NUMERIC_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?k?\b")


def _normalize_numeric(token: str) -> str:
    cleaned = token.strip().lower().replace(",", "").replace(" ", "")
    if cleaned.endswith("k"):
        try:
            return str(int(float(cleaned[:-1]) * 1000))
        except ValueError:
            return cleaned
    if cleaned.endswith(".00"):
        cleaned = cleaned[:-3]
    return cleaned


def _numeric_candidates(text: str) -> set[str]:
    cleaned = re.sub(r"[$€£\s]", "", text or "")
    return {_normalize_numeric(tok) for tok in _NUMERIC_RE.findall(cleaned)}


def salary_representations(value: int) -> set[str]:
    """Cosmetic forms a model might use for the same figure."""
    digits = str(value)
    return {
        digits,
        f"{value:,}",
        f"${digits}",
        f"${value:,}",
        f"{digits}.00",
        f"{value / 1000:g}k",
    }


def assert_absent_salary(text: str, value: int) -> None:
    """Fail if ``value`` appears in ``text`` in any common formatting.

    Catches ``95000``, ``95,000``, ``$95,000``, ``95 000``, ``95000.00`` and
    ``95k``. Spelled-out/rounded paraphrases are out of scope; the DB-role
    tests are the guarantee that the value was never readable at all.
    """
    lowered = (text or "").lower()
    for representation in salary_representations(value):
        assert representation.lower() not in lowered, (
            f"leaked salary representation {representation!r} in: {text!r}"
        )
    assert str(value) not in _numeric_candidates(lowered), (
        f"leaked normalized salary {value} in: {text!r}"
    )


def make_sql_state(task: str) -> SupervisorState:
    return SupervisorState(
        messages=[HumanMessage(content=task)],
        current_task=task,
    )


def make_config(permission_level: str = "general", user_id: str = "test-user-1") -> RunnableConfig:
    return {
        "configurable": {
            "permission_level": permission_level,
            "user_id": user_id,
            "thread_id": "llm-test-thread",
        }
    }


def _set_tool_func(tool, func) -> None:
    """StructuredTool is a Pydantic model; func is a real field, invoke is not."""
    try:
        tool.func = func
    except Exception:
        object.__setattr__(tool, "func", func)


class ToolCallRecorder:
    """
    Records tool invocations without changing production behavior.

    LangChain StructuredTool is a Pydantic model: instance `invoke` cannot be
    patched. The actual Python callable lives on `.func` (used by both
    `.invoke()` and `.run()`), so we wrap that instead.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._originals: list[tuple[Any, Any]] = []

    def track(self, *tools) -> "ToolCallRecorder":
        for tool in tools:
            original_func = tool.func
            name = tool.name

            def _make_wrapper(orig, tool_name):
                def wrapper(*args, **kwargs):
                    if kwargs:
                        payload = kwargs
                    elif len(args) == 1:
                        payload = args[0]
                    else:
                        payload = args
                    self.calls.append((tool_name, payload))
                    return orig(*args, **kwargs)

                return wrapper

            wrapper = _make_wrapper(original_func, name)
            self._originals.append((tool, original_func))
            _set_tool_func(tool, wrapper)
        return self

    def stop(self) -> None:
        for tool, original_func in self._originals:
            _set_tool_func(tool, original_func)
        self._originals.clear()

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def called(self, tool_name: str) -> bool:
        return tool_name in self.names()

    def __enter__(self) -> "ToolCallRecorder":
        return self

    def __exit__(self, *args) -> None:
        self.stop()