from __future__ import annotations

import time

import pytest
from langchain_core.messages import HumanMessage

from agents.supervisor import supervisor_agent
from agents.sql_agent import Sql_agent
from services.budget import TurnBudget, TurnBudgetExceeded, budget_scope, get_budget
from services.llm import ResilientLLM
from state.state import PlanItem, SpecialistResult, SupervisorState
from tests.conftest import make_config, make_sql_state


class _FakeResponse:
    def __init__(self, total_tokens: int | None = None):
        self.content = "ok"
        self.usage_metadata = (
            {"total_tokens": total_tokens} if total_tokens is not None else None
        )


class _FakeClient:
    def __init__(self, total_tokens: int | None = None):
        self.total_tokens = total_tokens

    def invoke(self, *args, **kwargs):
        return _FakeResponse(total_tokens=self.total_tokens)


def _fake_llm(total_tokens: int | None = None) -> ResilientLLM:
    return ResilientLLM([("fake", lambda: _FakeClient(total_tokens=total_tokens))])


def test_check_and_count_raises_at_cap():
    budget = TurnBudget(max_calls=2, max_tokens=10_000, max_seconds=10)
    budget.check_and_count()
    budget.check_and_count()
    with pytest.raises(TurnBudgetExceeded) as exc:
        budget.check_and_count()
    assert exc.value.kind == "llm_call"


def test_token_budget_exhausted():
    budget = TurnBudget(max_calls=10, max_tokens=100, max_seconds=10)
    budget.record_usage(_FakeResponse(total_tokens=100))
    assert budget.exhausted()
    with pytest.raises(TurnBudgetExceeded) as exc:
        budget.check_and_count()
    assert exc.value.kind == "token"


def test_llm_time_budget_ignores_idle_time():
    budget = TurnBudget(max_calls=10, max_tokens=10_000, max_seconds=1.0)
    time.sleep(0.05)  # human think-time / tool time must not count
    assert not budget.exhausted()


def test_llm_time_budget_exhausted_by_call_duration():
    budget = TurnBudget(max_calls=10, max_tokens=10_000, max_seconds=1.0)
    budget.record_duration(1.0)
    assert budget.exhausted()
    with pytest.raises(TurnBudgetExceeded) as exc:
        budget.check_and_count()
    assert exc.value.kind == "wall_clock"


class _RetryableError(Exception):
    status_code = 503


class _FailingClient:
    def invoke(self, *args, **kwargs):
        raise _RetryableError("service unavailable")


def test_budget_counts_each_provider_attempt():
    model = ResilientLLM(
        [
            ("provider-a", lambda: _FailingClient()),
            ("provider-b", lambda: _FailingClient()),
        ]
    )
    budget = TurnBudget(max_calls=1, max_tokens=10_000, max_seconds=10)
    with budget_scope(budget):
        with pytest.raises(TurnBudgetExceeded):
            model.invoke([HumanMessage(content="hi")])
    # The first fallback attempt consumed the single allowed call.
    assert budget.calls == 1


def test_failed_client_construction_does_not_consume_budget():
    def bad_factory():
        raise RuntimeError("missing api key")

    model = ResilientLLM([("a", bad_factory), ("b", bad_factory)])
    budget = TurnBudget(max_calls=5, max_tokens=10_000, max_seconds=10)
    with budget_scope(budget):
        with pytest.raises(RuntimeError):
            model.invoke([HumanMessage(content="hi")])

    assert budget.calls == 0


def test_exhausted_budget_short_circuits_before_build():
    built = {"n": 0}

    def factory():
        built["n"] += 1
        return _FakeClient()

    model = ResilientLLM([("a", factory)])
    with budget_scope(TurnBudget(max_calls=0, max_tokens=1, max_seconds=1)):
        with pytest.raises(TurnBudgetExceeded):
            model.invoke([HumanMessage(content="hi")])

    assert built["n"] == 0


def test_chart_route_survives_exhausted_budget():
    state = SupervisorState(
        messages=[HumanMessage(content="chart that")],
        turn_count=1,
        plan_ready=True,
        last_result=SpecialistResult(
            source="sql",
            summary="Sales by region",
            status="done",
            structured_data=[{"label": "MENA", "value": 1200.5}],
        ),
        plan=[
            PlanItem(
                route="sql",
                task="sales",
                status="done",
                structured_data=[{"label": "MENA", "value": 1200.5}],
            )
        ],
    )
    with budget_scope(TurnBudget(max_calls=0, max_tokens=1, max_seconds=1)):
        update = supervisor_agent(state, {"configurable": {}})
    assert update["next"] == "visu"


def test_no_scope_is_unlimited():
    assert get_budget() is None
    model = _fake_llm()
    for _ in range(50):
        assert model.invoke([HumanMessage(content="hi")]).content == "ok"


def test_scope_counts_calls_and_raises():
    with budget_scope(TurnBudget(max_calls=1, max_tokens=10_000, max_seconds=10)):
        model = _fake_llm()
        model.invoke([HumanMessage(content="hi")])
        with pytest.raises(TurnBudgetExceeded):
            model.invoke([HumanMessage(content="again")])


def test_scope_resets_after_exit():
    with budget_scope(TurnBudget(max_calls=0, max_tokens=1, max_seconds=1)):
        assert get_budget() is not None
    assert get_budget() is None


def test_supervisor_ends_when_budget_exhausted_before_planning():
    state = SupervisorState(messages=[HumanMessage(content="hello")])
    with budget_scope(TurnBudget(max_calls=0, max_tokens=1, max_seconds=1)):
        update = supervisor_agent(state, {"configurable": {}})
    assert update["next"] == "end"


def test_sql_agent_degrades_gracefully_on_budget_exceeded(monkeypatch):
    monkeypatch.setattr("agents.sql_agent.llm", lambda *a, **k: _fake_llm())
    with budget_scope(TurnBudget(max_calls=0, max_tokens=1, max_seconds=1)):
        result = Sql_agent(
            make_sql_state("How many employees are there?"),
            make_config("general"),
        )["last_result"]
    assert result.status == "failed"
    assert result.issue == "budget_exceeded"
