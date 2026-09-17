from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import StateSnapshot, Command, interrupt

from graph.workflow import has_pending_interrupt
from state.state import SupervisorState


def _snapshot(interrupts=()) -> StateSnapshot:
    return StateSnapshot(
        values={},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=interrupts,
    )


def test_no_pending_interrupt():
    assert has_pending_interrupt(_snapshot()) is False


def test_pending_interrupt_detected():
    assert has_pending_interrupt(_snapshot((object(),))) is True


def test_object_without_interrupts_attribute_is_false():
    assert has_pending_interrupt(object()) is False


def _pause_node(state):
    interrupt({"question": "which one?"})
    return {}


def test_real_interrupt_snapshot_then_resume():
    builder = StateGraph(SupervisorState)
    builder.add_node("pause", _pause_node)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", END)
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "interrupt-test"}}

    graph.invoke({"messages": []}, config)

    assert has_pending_interrupt(graph.get_state(config)) is True

    graph.invoke(Command(resume="the first one"), config)

    assert has_pending_interrupt(graph.get_state(config)) is False
