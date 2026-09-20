"""
Deterministic supervisor rules — no LLM.

Covers hop limits, done→end, auto-visu, permission_denied, RAG empty,
repeat-failure, post-guards, and duplicate-task blocking.
"""

from __future__ import annotations

from typing import Literal

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.supervisor import (
    MAX_HOPS_PER_TURN,
    MAX_PLAN_STEPS,
    deterministic_decision,
    get_supervisor_decision,
    post_decision_guards,
    supervisor_agent,
    _has_multi_intent,
    _maybe_insert_visu,
    _plan_from_workflow,
    _user_wants_visualization,
)
from state.structure_output import PlanStep, WorkflowPlan
from state.state import PlanItem, SpecialistResult, SupervisorState, TaskRecord
from state.structure_output import SupervisorDecision
from services.errors import LLMOutageError
from services.message_utils import CLARIFICATION_ANSWER_FLAG


def _record(
    *,
    turn: int = 1,
    route: Literal["rag", "convo", "sql", "research", "visu"] = "sql",
    task: str = "task",
    status: Literal["done", "partial", "failed"] = "done",
    summary: str = "ok",
    issue: str | None = None,
) -> TaskRecord:
    return TaskRecord(
        turn=turn,
        route=route,
        task=task,
        status=status,
        result_summary=summary,
        issue=issue,
    )


def _state(
    *,
    text: str = "hello",
    last_result: SpecialistResult | None = None,
    current_task: str | None = None,
    task_history: list[TaskRecord] | None = None,
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


def test_hop_limit_forces_end():
    history = [_record(task=f"t{i}") for i in range(MAX_HOPS_PER_TURN)]
    state = _state(last_result=_done_sql_with_rows(), current_task="t0", task_history=history)
    decision = deterministic_decision(state, history)
    assert decision is not None
    assert decision.next == "end"


def test_empty_messages_forces_end():
    state = _state(messages=[], last_result=None)
    decision = deterministic_decision(state, [])
    assert decision is not None
    assert decision.next == "end"


def test_no_last_result_falls_through_to_llm():
    state = _state(text="How many employees are there?", last_result=None)
    assert deterministic_decision(state, []) is None


def test_done_result_forces_end():
    lr = SpecialistResult(source="sql", summary="There are 2 employees.", status="done")
    state = _state(text="How many employees?", last_result=lr, current_task="count employees")
    decision = deterministic_decision(state, [])
    assert decision is not None
    assert decision.next == "end"


def test_done_with_structured_data_and_chart_intent_routes_visu():
    state = _state(
        text="Make a bar chart of that",
        last_result=_done_sql_with_rows(),
        current_task="sales by region",
    )
    decision = deterministic_decision(state, [])
    assert decision is not None
    assert decision.next == "visu"
    assert "chart" in decision.current_task.lower()


def test_done_with_structured_data_but_no_chart_intent_ends():
    state = _state(
        text="How many sales by region?",
        last_result=_done_sql_with_rows(),
        current_task="sales by region",
    )
    decision = deterministic_decision(state, [])
    assert decision is not None
    assert decision.next == "end"


def test_done_without_structured_data_does_not_auto_visu_even_with_chart_words():
    lr = SpecialistResult(source="sql", summary="2 employees", status="done")
    state = _state(text="Now chart that", last_result=lr)
    decision = deterministic_decision(state, [])
    assert decision is not None
    assert decision.next == "end"


def test_permission_denied_routes_convo_once():
    lr = SpecialistResult(
        source="sql",
        summary="Salary data requires elevated permissions.",
        status="failed",
        issue="permission_denied",
    )
    state = _state(text="Show salaries", last_result=lr, current_task="salaries")
    decision = deterministic_decision(state, [])
    assert decision is not None
    assert decision.next == "convo"
    assert "permission" in decision.current_task.lower()


def test_permission_denied_already_explained_ends():
    lr = SpecialistResult(
        source="sql",
        summary="denied",
        status="failed",
        issue="permission_denied",
    )
    history = [_record(route="convo", task="explain restriction", status="done")]
    state = _state(text="Show salaries", last_result=lr, task_history=history)
    decision = deterministic_decision(state, history)
    assert decision is not None
    assert decision.next == "end"


def test_rag_no_matching_documents_routes_convo():
    lr = SpecialistResult(
        source="rag",
        summary="No matching internal documents were found for this request.",
        status="partial",
        issue="no_matching_documents",
    )
    state = _state(text="Internal Antarctica policy?", last_result=lr)
    decision = deterministic_decision(state, [])
    assert decision is not None
    assert decision.next == "convo"
    assert "internal" in decision.current_task.lower() or "document" in decision.current_task.lower()


def test_same_specialist_already_failed_routes_convo():
    lr = SpecialistResult(source="sql", summary="could not finish", status="failed", issue="loop")
    history = [
        _record(route="sql", task="query x", status="failed", issue="loop"),
        _record(route="sql", task="query x", status="failed", issue="loop"),
    ]
    state = _state(text="try again", last_result=lr, current_task="query x", task_history=history)
    decision = deterministic_decision(state, history)
    assert decision is not None
    assert decision.next == "convo"


def test_first_failure_allows_one_retry_residual():
    lr = SpecialistResult(
        source="sql", summary="could not finish", status="failed", issue="invalid_request"
    )
    history = [_record(route="sql", task="query x", status="failed", issue="invalid_request")]
    state = _state(text="try again", last_result=lr, current_task="query x", task_history=history)
    assert deterministic_decision(state, history) is None


def test_repeat_failure_convo_task_has_safe_reason():
    lr = SpecialistResult(
        source="sql",
        summary="The requested data could not be found in the database.",
        status="failed",
        issue='table_or_schema_missing: relation "x" does not exist',
    )
    history = [
        _record(route="sql", task="query x", status="failed", issue="loop"),
        _record(route="sql", task="query x", status="failed", issue="loop"),
    ]
    state = _state(text="try again", last_result=lr, current_task="query x", task_history=history)
    decision = deterministic_decision(state, history)
    assert decision is not None
    assert decision.next == "convo"
    assert "could not be found" in decision.current_task
    assert "does not exist" not in decision.current_task


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


def test_multi_intent_detected_on_two_question_clauses():
    state = SupervisorState(
        messages=[
            HumanMessage(
                content="How many employees are there, and what does the policy say?"
            )
        ]
    )
    assert _has_multi_intent(state)


def test_two_questions_without_separator_are_multi_intent():
    state = SupervisorState(
        messages=[HumanMessage(content="How many employees? What does the policy say?")]
    )
    assert _has_multi_intent(state)


def test_single_intent_conjunction_is_not_multi_intent():
    state = SupervisorState(
        messages=[HumanMessage(content="Show employees and departments.")]
    )
    assert not _has_multi_intent(state)


def test_multi_intent_done_allows_residual_when_hop_available():
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
        multi_intent_hops=0,
    )
    assert deterministic_decision(state, []) is None


def test_multi_intent_done_ends_when_hop_used():
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
        multi_intent_hops=1,
    )
    decision = deterministic_decision(state, [])
    assert decision is not None
    assert decision.next == "end"


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

    with pytest.raises(LLMOutageError):
        get_supervisor_decision(context, _Model())


def test_same_route_cap_forces_end():
    history = [
        _record(route="sql", task="a"),
        _record(route="sql", task="b"),
    ]
    state = _state(text="again", task_history=history)
    decision = SupervisorDecision(next="sql", current_task="c")
    guarded = post_decision_guards(state, decision, history)
    assert guarded.next == "end"


def test_refuse_reroute_to_just_finished_specialist():
    lr = SpecialistResult(source="rag", summary="policy text", status="done")
    state = _state(text="thanks", last_result=lr)
    decision = SupervisorDecision(next="rag", current_task="repeat")
    guarded = post_decision_guards(state, decision, [])
    assert guarded.next == "end"


def test_first_clarification_in_same_turn_is_allowed():
    state = _state(text="still unclear", clarification_count=0)
    decision = SupervisorDecision(next="clarification", current_task="ask once")
    guarded = post_decision_guards(state, decision, [])
    assert guarded.next == "clarification"


def test_clarification_cap_routes_to_convo():
    state = _state(text="still unclear", clarification_count=1)
    decision = SupervisorDecision(next="clarification", current_task="ask again")
    guarded = post_decision_guards(state, decision, [])
    assert guarded.next == "convo"


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
    plan = [PlanItem(route="sql", task="sales", status="done", structured_data=[{"label": "a", "value": 1}])]

    updated = _maybe_insert_visu(plan, state)

    assert [item.route for item in updated] == ["sql", "visu"]


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