"""Supervisor rules — plan-based, no LLM.

Covers chart-intent detection, workflow-plan construction, deterministic plan
execution, the single-decision fallback, and the clarification cap.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agents.supervisor import (
    MAX_PLAN_STEPS,
    get_supervisor_decision,
    supervisor_agent,
    _apply_clarification_cap,
    _maybe_insert_visu,
    _plan_from_workflow,
    _user_wants_visualization,
)
from state.state import PlanItem, SpecialistResult, SupervisorState
from state.structure_output import PlanStep, WorkflowPlan
from services.errors import LLMOutageError
from services.message_utils import CLARIFICATION_ANSWER_FLAG


def _state(
    *,
    text: str = "hello",
    last_result: SpecialistResult | None = None,
    current_task: str | None = None,
    task_history=None,
    turn_count: int = 1,
    clarification_count: int = 0,
    messages=None,
) -> SupervisorState:
    return SupervisorState(
        messages=messages if messages is not None else [HumanMessage(content=text)],
        last_result=last_result,
        current_task=current_task,
        task_history=task_history or [],
        turn_count=turn_count,
        clarification_count=clarification_count,
    )


def _done_sql_with_rows() -> SpecialistResult:
    return SpecialistResult(
        source="sql",
        summary="Sales by region",
        status="done",
        structured_data=[{"label": "MENA", "value": 1200.5}, {"label": "EU", "value": 800.0}],
    )


# --- chart intent -----------------------------------------------------------


def test_user_wants_visualization_keywords():
    for text in (
        "chart this",
        "make a graph",
        "plot sales",
        "visualize it",
        "visualise the data",
        "bar chart please",
        "pie of departments",
        "line chart of sales",
    ):
        assert _user_wants_visualization(_state(text=text)), text


def test_user_does_not_want_visualization_on_plain_data_question():
    assert not _user_wants_visualization(_state(text="How many employees are there?"))


def test_user_wants_visualization_new_chart_types():
    for text in (
        "Plot a histogram of sales",
        "Make a scatter plot",
        "Create a donut chart",
        "show a doughnut graph",
    ):
        assert _user_wants_visualization(_state(text=text)), text


def test_chart_vocabulary_collocations_are_not_visualization():
    for text in (
        "Show me the org chart.",
        "What is the chart of accounts?",
        "He is a member of the bar association.",
        "She passed the bar exam.",
        "We bought a plot of land.",
        "This is a graph database.",
        "Graph theory is interesting.",
        "I entered a pie eating contest.",
    ):
        assert not _user_wants_visualization(_state(text=text)), text


def test_user_wants_visualization_multimodal_content():
    state = SupervisorState(
        messages=[HumanMessage(content=[{"type": "text", "text": "Please visualize this"}])],
        turn_count=1,
    )
    assert _user_wants_visualization(state)


def test_user_wants_visualization_from_clarification_answer():
    state = SupervisorState(
        messages=[
            HumanMessage(content="show me the sales numbers"),
            AIMessage(content="Which view would you like?"),
            HumanMessage(
                content="as a pie chart",
                additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
            ),
        ],
        turn_count=1,
    )

    assert _user_wants_visualization(state)


def test_intent_from_bare_pie_answer():
    state = SupervisorState(
        messages=[
            HumanMessage(content="show me the sales numbers"),
            AIMessage(content="Which view would you like?"),
            HumanMessage(
                content="as a pie",
                additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
            ),
        ],
        turn_count=1,
    )

    assert _user_wants_visualization(state)


def test_stale_clarification_answer_does_not_trigger_visualization():
    state = SupervisorState(
        messages=[
            HumanMessage(content="show me a chart of sales"),
            AIMessage(content="Which view would you like?"),
            HumanMessage(
                content="pie chart",
                additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
            ),
            HumanMessage(content="what were total sales last quarter?"),
        ],
        turn_count=2,
    )

    assert not _user_wants_visualization(state)


def test_intent_ignores_bar_substring():
    state = SupervisorState(
        messages=[HumanMessage(content="How is the Barcelona office doing?")],
        turn_count=1,
    )

    assert not _user_wants_visualization(state)


# --- workflow plan construction --------------------------------------------


def test_plan_from_workflow_builds_ordered_steps():
    workflow = WorkflowPlan(
        steps=[
            PlanStep(route="sql", task="count employees"),
            PlanStep(route="rag", task="remote work policy"),
        ]
    )
    plan = _plan_from_workflow(workflow)
    assert [item.route for item in plan] == ["sql", "rag"]
    assert all(item.status == "pending" for item in plan)


def test_plan_from_workflow_clarification_is_alone():
    workflow = WorkflowPlan(
        steps=[
            PlanStep(route="clarification", task="Which report?"),
            PlanStep(route="sql", task="count"),
        ]
    )
    plan = _plan_from_workflow(workflow)
    assert [item.route for item in plan] == ["clarification"]


def test_plan_from_workflow_caps_steps():
    workflow = WorkflowPlan(
        steps=[
            PlanStep(route="sql", task=f"task {i}") for i in range(MAX_PLAN_STEPS + 2)
        ]
    )
    assert len(_plan_from_workflow(workflow)) == MAX_PLAN_STEPS


def test_maybe_insert_visu_adds_step_after_sql_rows():
    state = _state(text="Make a bar chart of that", turn_count=1)
    plan = [
        PlanItem(
            route="sql",
            task="sales",
            status="done",
            structured_data=[{"label": "a", "value": 1}],
        )
    ]

    updated = _maybe_insert_visu(plan, state)

    assert [item.route for item in updated] == ["sql", "visu"]


def test_clarification_cap_converts_to_convo():
    state = _state(text="unclear", clarification_count=1)
    plan = [PlanItem(route="clarification", task="which one?")]

    updated = _apply_clarification_cap(plan, state)

    assert updated[0].route == "convo"


# --- supervisor_agent -------------------------------------------------------


def test_routing_failure_sensitive_routes_to_sql(monkeypatch):
    monkeypatch.setattr("agents.supervisor.get_workflow_plan", lambda *a, **k: [])
    monkeypatch.setattr("agents.supervisor.get_supervisor_decision", lambda *a, **k: None)
    state = _state(text="What is Rami Noueihed's salary?", turn_count=1)

    update = supervisor_agent(state, {"configurable": {}})

    assert update["next"] == "sql"
    assert "salary" in update["current_task"].lower()


def test_routing_failure_non_sensitive_routes_to_convo(monkeypatch):
    monkeypatch.setattr("agents.supervisor.get_workflow_plan", lambda *a, **k: [])
    monkeypatch.setattr("agents.supervisor.get_supervisor_decision", lambda *a, **k: None)
    state = _state(text="Tell me something interesting.", turn_count=1)

    update = supervisor_agent(state, {"configurable": {}})

    assert update["next"] == "convo"


def test_supervisor_agent_advances_plan_deterministically(monkeypatch):
    """After the first step completes, the next planned step is forced without
    any further LLM routing call."""
    def boom(*_a, **_k):
        raise AssertionError("no LLM routing call should happen mid-plan")

    monkeypatch.setattr("agents.supervisor.llm", boom)

    lr = SpecialistResult(source="sql", summary="2 employees", status="done")
    state = SupervisorState(
        messages=[
            HumanMessage(
                content="How many employees are there, and what does the policy say?"
            )
        ],
        last_result=lr,
        current_task="count employees",
        turn_count=1,
        plan_ready=True,
        plan=[
            PlanItem(route="sql", task="count employees", status="pending"),
            PlanItem(route="rag", task="remote work policy", status="pending"),
        ],
    )

    update = supervisor_agent(state, {"configurable": {}})

    assert update["next"] == "rag"
    assert update["plan"][0].status == "done"
    assert update["plan"][1].status == "pending"


def test_llm_outage_ends_turn_deterministically(monkeypatch):
    def outage(*_a, **_k):
        raise LLMOutageError("transient")

    monkeypatch.setattr("agents.supervisor.get_workflow_plan", outage)
    state = _state(text="Tell me something interesting.", turn_count=1)

    update = supervisor_agent(state, {"configurable": {}})

    assert update["next"] == "end"
    assert update["outage"] is True


def test_get_supervisor_decision_raises_on_auth_error():
    class _AuthError(Exception):
        status_code = 401

    class _Structured:
        def invoke(self, _prompt):
            raise _AuthError("unauthorized")

    class _Model:
        def with_structured_output(self, _schema, **_kwargs):
            return _Structured()

    context = {
        "messages": [],
        "last_result": None,
        "task_history": [],
        "known_facts": "",
        "conversation_summary": None,
    }

    import pytest

    with pytest.raises(LLMOutageError):
        get_supervisor_decision(context, _Model())


def test_supervisor_agent_skips_llm_when_last_result_done(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("LLM must not be called on a done last_result")

    monkeypatch.setattr("agents.supervisor.llm", boom)

    lr = SpecialistResult(source="sql", summary="2 employees", status="done")
    state = _state(
        text="How many employees?",
        last_result=lr,
        current_task="count employees",
        turn_count=1,
    )
    state.plan_ready = True
    state.plan = [PlanItem(route="sql", task="count employees", status="pending")]

    update = supervisor_agent(state, {"configurable": {}})
    assert update["next"] == "end"
    assert len(update["task_history"]) == 1
    assert update["task_history"][0].route == "sql"


def test_supervisor_agent_auto_visu_skips_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("LLM must not be called for auto-visu")

    monkeypatch.setattr("agents.supervisor.llm", boom)

    state = _state(
        text="Make a bar chart of that",
        last_result=_done_sql_with_rows(),
        current_task="sales by region",
        turn_count=1,
    )
    state.plan_ready = True
    state.plan = [PlanItem(route="sql", task="sales by region", status="pending")]

    update = supervisor_agent(state, {"configurable": {}})
    assert update["next"] == "visu"
    assert "last_result" not in update
