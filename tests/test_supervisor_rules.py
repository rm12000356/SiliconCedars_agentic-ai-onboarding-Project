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
    _skip_unbudgeted_visu,
    _user_wants_visualization,
)
from state.state import PlanItem, SpecialistResult, SupervisorState
from state.structure_output import PlanStep, WorkflowPlan
from services.budget import TurnBudget, budget_scope
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


def test_plan_from_workflow_carries_visu_data_source():
    workflow = WorkflowPlan(
        steps=[
            PlanStep(route="sql", task="sales"),
            PlanStep(
                route="visu",
                task="pie chart: 60 EU, 25 MENA, 15 APAC",
                data_source="inline",
            ),
        ]
    )

    plan = _plan_from_workflow(workflow)

    assert plan[1].data_source == "inline"


def test_plan_from_workflow_infers_visu_data_source():
    after_sql = _plan_from_workflow(
        WorkflowPlan(
            steps=[
                PlanStep(route="sql", task="sales"),
                PlanStep(route="visu", task="chart sales"),
            ]
        )
    )
    standalone = _plan_from_workflow(
        WorkflowPlan(steps=[PlanStep(route="visu", task="chart 1, 2, 3")])
    )

    assert after_sql[1].data_source == "database"
    assert standalone[0].data_source == "inline"


def test_maybe_insert_visu_adds_step_after_sql_rows():
    state = _state(
        text="Make a bar chart of that",
        last_result=_done_sql_with_rows(),
        turn_count=1,
    )
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
    assert updated[-1].data_source == "database"


def test_maybe_insert_visu_not_reinserted_after_done_visu():
    state = _state(
        text="Make a bar chart of that",
        last_result=SpecialistResult(
            source="visu", summary="Here's the chart.", status="done"
        ),
        turn_count=1,
    )
    plan = [
        PlanItem(
            route="sql",
            task="sales",
            status="done",
            structured_data=[{"label": "a", "value": 1}],
        ),
        PlanItem(route="visu", task="chart", status="done"),
    ]

    updated = _maybe_insert_visu(plan, state)

    assert [item.route for item in updated] == ["sql", "visu"]


def test_maybe_insert_visu_no_insert_without_structured_data():
    state = _state(
        text="Make a bar chart of that",
        last_result=SpecialistResult(source="sql", summary="no rows", status="done"),
        turn_count=1,
    )
    plan = [PlanItem(route="sql", task="sales", status="done")]

    updated = _maybe_insert_visu(plan, state)

    assert [item.route for item in updated] == ["sql"]


def test_skip_unbudgeted_visu_blocks_llm_dependent_visu():
    state = _state(
        text="Make a bar chart of that",
        last_result=SpecialistResult(source="sql", summary="no rows", status="done"),
        turn_count=1,
    )
    plan = [PlanItem(route="visu", task="inline chart")]

    with budget_scope(TurnBudget(max_calls=0, max_tokens=0, max_seconds=0.0)):
        updated = _skip_unbudgeted_visu(plan, state)

    assert updated[0].status == "failed"
    assert updated[0].issue == "budget_exceeded"
    assert updated[0].result_summary


def test_skip_unbudgeted_visu_blocks_inline_visu_even_with_rows():
    state = _state(
        text="Make a bar chart of that",
        last_result=_done_sql_with_rows(),
        turn_count=1,
    )
    plan = [PlanItem(route="visu", task="inline chart", data_source="inline")]

    with budget_scope(TurnBudget(max_calls=0, max_tokens=0, max_seconds=0.0)):
        updated = _skip_unbudgeted_visu(plan, state)

    assert updated[0].status == "failed"
    assert updated[0].issue == "budget_exceeded"


def test_skip_unbudgeted_visu_allows_deterministic_visu():
    state = _state(
        text="Make a bar chart of that",
        last_result=_done_sql_with_rows(),
        turn_count=1,
    )
    plan = [PlanItem(route="visu", task="chart")]

    with budget_scope(TurnBudget(max_calls=0, max_tokens=0, max_seconds=0.0)):
        updated = _skip_unbudgeted_visu(plan, state)

    assert updated[0].status == "pending"


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


def test_supervisor_does_not_loop_back_to_visu_after_done(monkeypatch):
    """Once a visu step has run, the supervisor must terminate instead of
    appending another visu (which would fall back to an LLM call and loop)."""
    def boom(*_a, **_k):
        raise AssertionError("LLM must not be called after a completed visu")

    monkeypatch.setattr("agents.supervisor.llm", boom)

    state = _state(
        text="Make a bar chart of that",
        last_result=SpecialistResult(
            source="visu", summary="Here's the chart.", status="done"
        ),
        current_task="Create a clear chart from the structured_data.",
        turn_count=1,
    )
    state.plan_ready = True
    state.plan = [
        PlanItem(
            route="sql",
            task="sales",
            status="done",
            structured_data=[{"label": "a", "value": 1}],
        ),
        PlanItem(route="visu", task="chart", status="pending"),
    ]

    update = supervisor_agent(state, {"configurable": {}})

    assert update["next"] == "end"
    assert [item.route for item in update["plan"]] == ["sql", "visu"]


def test_supervisor_skips_llm_visu_when_budget_exhausted(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("LLM must not be called when budget is exhausted")

    monkeypatch.setattr("agents.supervisor.llm", boom)

    state = _state(
        text="Make a bar chart of that",
        last_result=SpecialistResult(source="sql", summary="no rows", status="failed"),
        current_task="sales by region",
        turn_count=1,
    )
    state.plan_ready = True
    state.plan = [
        PlanItem(route="sql", task="sales by region", status="pending"),
        PlanItem(route="visu", task="chart"),
    ]

    with budget_scope(TurnBudget(max_calls=0, max_tokens=0, max_seconds=0.0)):
        update = supervisor_agent(state, {"configurable": {}})

    assert update["next"] == "end"
    assert update["plan"][1].status == "failed"
    assert update["plan"][1].issue == "budget_exceeded"


def test_supervisor_clears_last_result_for_inline_visu(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("no LLM call expected while routing to visu")

    monkeypatch.setattr("agents.supervisor.llm", boom)

    state = _state(
        text="Count employees and make a pie chart: 60 EU, 40 US",
        last_result=_done_sql_with_rows(),
        current_task="count employees",
        turn_count=1,
    )
    state.plan_ready = True
    state.plan = [
        PlanItem(route="sql", task="count employees", status="pending"),
        PlanItem(
            route="visu",
            task="pie chart: 60 EU, 40 US",
            data_source="inline",
        ),
    ]

    update = supervisor_agent(state, {"configurable": {}})

    assert update["next"] == "visu"
    assert update["last_result"] is None


def test_supervisor_preserves_last_result_for_database_visu(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("no LLM call expected while routing to visu")

    monkeypatch.setattr("agents.supervisor.llm", boom)

    state = _state(
        text="Make a bar chart of that",
        last_result=_done_sql_with_rows(),
        current_task="count employees",
        turn_count=1,
    )
    state.plan_ready = True
    state.plan = [
        PlanItem(route="sql", task="count employees", status="pending"),
        PlanItem(route="visu", task="chart", data_source="database"),
    ]

    update = supervisor_agent(state, {"configurable": {}})

    assert update["next"] == "visu"
    assert "last_result" not in update
