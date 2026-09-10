from __future__ import annotations

import os
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

        with get_general_connection() as conn:
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