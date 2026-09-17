from __future__ import annotations

import os
import uuid

import pytest
from langgraph.graph import StateGraph, START, END

from services.memory import get_checkpointer
from state.state import SpecialistResult, SupervisorState, TaskRecord
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]


def _write_node(state: SupervisorState) -> dict:
    return {
        "task_history": [
            TaskRecord(
                turn=1,
                route="sql",
                task="count employees",
                status="done",
                result_summary="2",
            )
        ],
        "last_result": SpecialistResult(source="sql", summary="2", status="done"),
    }


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL required")
def test_postgres_checkpointer_roundtrips_custom_models(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_BACKEND", "postgres")

    memory, saver_context = get_checkpointer()
    thread_id = f"qa-pg-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        builder = StateGraph(SupervisorState)
        builder.add_node("write", _write_node)
        builder.add_edge(START, "write")
        builder.add_edge("write", END)
        graph = builder.compile(checkpointer=memory)

        graph.invoke({"messages": []}, config)

        values = graph.get_state(config).values
        record = values["task_history"][0]
        result = values["last_result"]

        assert isinstance(record, TaskRecord)
        assert isinstance(result, SpecialistResult)
        assert record.route == "sql"
        assert result.summary == "2"
    finally:
        try:
            memory.delete_thread(thread_id)
        except Exception:
            pass
        if saver_context is not None:
            saver_context.__exit__(None, None, None)
