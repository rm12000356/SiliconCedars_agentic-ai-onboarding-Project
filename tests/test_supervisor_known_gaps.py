"""
Known supervisor / subgraph gaps from the QA report.

xfail(strict=True) = current code is wrong; when you fix production,
the test will XPASS and you must drop the mark.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from agents.finalize import Finalize
from agents.research.supervisor import MAX_RESEARCH_ATTEMPTS, Sub_controler
from agents.supervisor import deterministic_decision, post_decision_guards, supervisor_agent
from state.state import SpecialistResult, SubGraphSupervisorState, SupervisorState, TaskRecord
from state.structure_output import SupervisorDecision


CHARTABLE = [
    {"label": "MENA", "value": 1200.5},
    {"label": "EU", "value": 800.0},
]


def _sql_done() -> SpecialistResult:
    return SpecialistResult(
        source="sql",
        summary="MENA 1200.5, EU 800.0",
        status="done",
        structured_data=CHARTABLE,
    )


@pytest.mark.xfail(
    reason="clarification is never written to task_history (not a SpecialistRoute), so the one-clarification guard never fires",
    strict=True,
)
def test_repeated_clarification_should_be_capped():
    """After a real clarification hop, supervisor_agent records nothing."""
    state = SupervisorState(
        messages=[HumanMessage(content="do the thing")],
        last_result=None,
        current_task=None,
        task_history=[],
        turn_count=1,
    )
    decision = SupervisorDecision(next="clarification", current_task="please clarify")
    guarded = post_decision_guards(state, decision, list(state.task_history))
    assert guarded.next == "end"


@pytest.mark.xfail(
    reason="viz keyword 'bar' matches unrelated wording and can force auto-visu",
    strict=True,
)
def test_chocolate_bar_does_not_auto_visu_after_sql():
    state = SupervisorState(
        messages=[HumanMessage(content="What is our policy on chocolate bars?")],
        last_result=_sql_done(),
        current_task="sales by region",
        turn_count=1,
    )
    decision = deterministic_decision(state, [])
    assert decision is not None
    assert decision.next != "visu"


@pytest.mark.xfail(
    reason="Finalize clears last_result; second user turn cannot auto-visu",
    strict=True,
    raises=AssertionError,
)
def test_second_user_message_chart_that_should_auto_visu(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("should be deterministic auto-visu, not LLM")

    monkeypatch.setattr("agents.supervisor.llm", boom)

    state = SupervisorState(
        messages=[
            HumanMessage(content="Show sales by region."),
            HumanMessage(content="Now chart that"),
        ],
        last_result=None,
        current_task=None,
        task_history=[
            TaskRecord(
                turn=1,
                route="sql",
                task="sales by region",
                status="done",
                result_summary="MENA 1200.5, EU 800.0",
            )
        ],
        turn_count=2,
    )
    update = supervisor_agent(state, {"configurable": {}})
    assert update["next"] == "visu"


def test_research_at_max_attempts_should_force_report_without_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("LLM must not be consulted at the attempt ceiling")

    monkeypatch.setattr("agents.research.supervisor.llm", boom)
    state = SubGraphSupervisorState(
        messages=[],
        task="anything",
        research_attempts=MAX_RESEARCH_ATTEMPTS,
        report_written=False,
    )
    result = Sub_controler(state)
    assert result == {"next": "report"}


def test_research_above_max_attempts_forces_report_without_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("LLM must not be consulted past the attempt ceiling")

    monkeypatch.setattr("agents.research.supervisor.llm", boom)
    state = SubGraphSupervisorState(
        messages=[],
        task="anything",
        research_attempts=MAX_RESEARCH_ATTEMPTS + 1,
        report_written=False,
    )
    result = Sub_controler(state)
    assert result == {"next": "report"}


def test_finalize_clears_last_result():
    state = SupervisorState(
        messages=[HumanMessage(content="How many employees?")],
        last_result=SpecialistResult(
            source="sql",
            summary="2 employees",
            status="done",
            structured_data=None,
        ),
        turn_count=1,
    )
    update = Finalize(state, {"configurable": {}})
    assert update.get("last_result") is None
    assert "messages" in update


def test_same_turn_sql_then_visu_still_works(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("LLM must not run on auto-visu")

    monkeypatch.setattr("agents.supervisor.llm", boom)
    state = SupervisorState(
        messages=[HumanMessage(content="Make a bar chart of sales by region.")],
        last_result=_sql_done(),
        current_task="sales by region",
        turn_count=1,
    )
    update = supervisor_agent(state, {"configurable": {}})
    assert update["next"] == "visu"