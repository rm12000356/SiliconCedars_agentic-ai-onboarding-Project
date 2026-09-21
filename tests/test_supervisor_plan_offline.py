"""Offline invariants for the supervisor's deterministic plan execution.

Runs the real `supervisor_agent` over seeded-random plans with stubbed
specialists and no LLM/DB access. Catches the dependent-step gap (a visu with
no data) and the routing-loop class of bug.
"""

from __future__ import annotations

import random

from langchain_core.messages import AIMessage, HumanMessage

from agents.supervisor import (
    MAX_HOPS,
    _next_pending,
    _rows_upstream,
    supervisor_agent,
)
from state.state import PlanItem, SpecialistResult, SupervisorState
from state.structure_output import SupervisorDecision
from services.message_utils import CLARIFICATION_ANSWER_FLAG

ROUTES = ["sql", "rag", "research", "visu", "convo", "clarification"]
PROMPT = (
    "Count employees, summarize the policy, research the current CEO, "
    "and chart the 2024 sales by region: EU 10 and MENA 20."
)


def _random_raw_plan(rng: random.Random) -> list[PlanItem]:
    plan: list[PlanItem] = []
    for _ in range(rng.randint(0, 6)):
        route = rng.choice(ROUTES)
        task = f"{route} ask {rng.randint(0, 999)}"
        if route == "visu":
            plan.append(
                PlanItem(
                    route="visu",
                    task=task,
                    data_source=rng.choice(["database", "inline", None]),
                )
            )
        else:
            plan.append(PlanItem(route=route, task=task))
    if rng.random() < 0.15:
        plan.append(PlanItem(route="clarification", task="which one?"))
    return plan


def _stub_result(route: str, rng: random.Random) -> SpecialistResult:
    if route == "sql":
        roll = rng.random()
        if roll < 0.4:
            return SpecialistResult(
                source="sql",
                summary="rows",
                status="done",
                structured_data=[{"label": "a", "value": 1}, {"label": "b", "value": 2}],
            )
        if roll < 0.7:
            return SpecialistResult(source="sql", summary="no rows", status="done")
        return SpecialistResult(
            source="sql", summary="db error", status="failed", issue="invalid_request"
        )
    return SpecialistResult(source=route, summary=f"{route} done", status="done")


def _resume_clarification(state: SupervisorState, update: dict) -> SupervisorState:
    plan = [item.model_copy() for item in update["plan"]]
    for item in plan:
        if item.status == "pending" and item.route == "clarification":
            item.status = "done"
            break
    return SupervisorState(
        messages=[
            *state.messages,
            AIMessage(content=state.clarification_question or "Which one?"),
            HumanMessage(
                content="the report",
                additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
            ),
        ],
        current_task=None,
        last_result=None,
        task_history=update.get("task_history", state.task_history),
        plan=plan,
        plan_ready=False,
        turn_count=state.turn_count,
        clarification_question=None,
        clarification_count=state.clarification_count + 1,
        conversation_summary=state.conversation_summary,
        chart_path=state.chart_path,
        hops=update.get("hops", state.hops),
        plan_note=update.get("plan_note", state.plan_note),
        turn_cut_short=update.get("turn_cut_short", state.turn_cut_short),
    )


def _advance(
    state: SupervisorState, update: dict, result: SpecialistResult
) -> SupervisorState:
    return SupervisorState(
        messages=state.messages,
        next=update.get("next"),
        current_task=update.get("current_task"),
        last_result=result,
        task_history=update.get("task_history", state.task_history),
        plan=update.get("plan", state.plan),
        plan_ready=True,
        turn_count=state.turn_count,
        clarification_count=state.clarification_count,
        conversation_summary=state.conversation_summary,
        chart_path=state.chart_path,
        hops=update.get("hops", state.hops),
        plan_note=update.get("plan_note", state.plan_note),
        turn_cut_short=update.get("turn_cut_short", state.turn_cut_short),
    )


class _BoomLLM:
    def with_structured_output(self, *a, **k):
        raise AssertionError("no LLM in offline test")

    def invoke(self, *a, **k):
        raise AssertionError("no LLM in offline test")


def _run_turn(monkeypatch, seed: int) -> dict:
    rng = random.Random(seed)
    raw_plan = _random_raw_plan(rng)

    monkeypatch.setattr(
        "agents.supervisor.get_workflow_plan",
        lambda *a, **k: [item.model_copy() for item in raw_plan],
    )
    monkeypatch.setattr(
        "agents.supervisor.get_supervisor_decision",
        lambda *a, **k: SupervisorDecision(next="convo", current_task="fallback"),
    )
    monkeypatch.setattr("agents.supervisor.llm", lambda *a, **k: _BoomLLM())

    state = SupervisorState(
        messages=[HumanMessage(content=PROMPT)], turn_count=1
    )
    routed_indices: list[int] = []
    last_update: dict = {}

    for _ in range(MAX_HOPS + 3):
        update = supervisor_agent(state, {"configurable": {}})
        last_update = update
        plan = update["plan"]
        nxt = update["next"]

        if nxt == "end":
            break

        item = _next_pending(plan)
        assert item is not None, "routing a step when no step is pending"
        index = next((j for j, p in enumerate(plan) if p is item), None)
        assert index is not None
        assert index not in routed_indices, f"step {index} routed more than once"
        routed_indices.append(index)

        if nxt == "visu" and item.data_source != "inline":
            assert _rows_upstream(plan, index) is not None, (
                "routed a database visu without upstream rows"
            )

        if nxt == "clarification":
            state = _resume_clarification(state, update)
            routed_indices.clear()  # a fresh plan is produced on resume
            continue

        state = _advance(state, update, _stub_result(nxt, rng))
    else:
        raise AssertionError("turn did not terminate within the hop cap")

    plan = last_update["plan"]
    assert last_update["next"] == "end"
    assert last_update["hops"] <= MAX_HOPS
    assert all(item.status != "pending" for item in plan), "a plan step was left pending"

    for index, item in enumerate(plan):
        if item.status == "skipped":
            assert _rows_upstream(plan, index) is None, (
                "skipped a step that had upstream data"
            )

    return last_update


def test_offline_plan_invariants(monkeypatch):
    for seed in range(60):
        _run_turn(monkeypatch, seed)
