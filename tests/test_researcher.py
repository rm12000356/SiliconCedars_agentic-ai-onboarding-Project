"""Unit tests for the research subgraph node — no LLM, no network.

The model and the tool table are stubbed so the deterministic control flow
(tool caps, unknown tools, forced final note, budget propagation) can be
exercised in isolation.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

import agents.research.researcher as researcher
from agents.research.researcher import (
    MAX_FETCH_ATTEMPTS,
    MAX_ITERATIONS,
    MAX_SEARCH_ATTEMPTS,
    Research,
)
from services.budget import TurnBudgetExceeded
from state.state import SubGraphSupervisorState


class _FakeTool:
    def __init__(self, name: str, result: str = "tool-result") -> None:
        self.name = name
        self.result = result
        self.calls: list[dict] = []

    def invoke(self, args):
        self.calls.append(args)
        return self.result


class _ScriptedModel:
    """Returns whatever ``script(messages)`` yields; ``bind_tools`` is a no-op."""

    def __init__(self, script):
        self._script = script
        self.calls = 0

    def bind_tools(self, *_a, **_k):
        return self

    def invoke(self, messages):
        self.calls += 1
        return self._script(messages)


def _tool_call(name: str, args=None, call_id: str = "1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id}],
    )


def _state(**overrides) -> SubGraphSupervisorState:
    values = {"messages": [], "task": "find something"}
    values.update(overrides)
    return SubGraphSupervisorState(**values)


def _patch(monkeypatch, script, tools):
    model = _ScriptedModel(script)
    monkeypatch.setattr(researcher, "llm", lambda *a, **k: model)
    monkeypatch.setattr(
        researcher, "TOOLS_BY_NAME", {t.name: t for t in tools}
    )
    return model


def test_research_empty_task_raises():
    try:
        Research(_state(task=""))
    except RuntimeError as exc:
        assert "empty task" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for empty task")


def test_research_returns_answer_when_no_tool_calls(monkeypatch):
    _patch(monkeypatch, lambda _m: AIMessage(content="final note"), [])

    result = Research(_state())

    assert result["research_messages"][0].content == "final note"


def test_research_caps_search_attempts(monkeypatch):
    search = _FakeTool("web_search")
    fetch = _FakeTool("fetch_page")

    def _script(messages):
        if "No more tool calls" in str(messages[-1].content):
            return AIMessage(content="final note")
        return _tool_call("web_search", {"query": "q"})

    _patch(monkeypatch, _script, [search, fetch])

    result = Research(_state())

    assert len(search.calls) == MAX_SEARCH_ATTEMPTS
    assert result["research_messages"][0].content == "final note"


def test_research_caps_fetch_attempts(monkeypatch):
    search = _FakeTool("web_search")
    fetch = _FakeTool("fetch_page")

    def _script(messages):
        if "No more tool calls" in str(messages[-1].content):
            return AIMessage(content="final note")
        return _tool_call("fetch_page", {"url": "https://e.com", "snippet": "s"})

    _patch(monkeypatch, _script, [search, fetch])

    Research(_state())

    assert len(fetch.calls) == MAX_FETCH_ATTEMPTS


def test_research_unknown_tool_is_reported_not_crashed(monkeypatch):
    def _script(messages):
        if "No more tool calls" in str(messages[-1].content):
            return AIMessage(content="final note")
        return _tool_call("bogus")

    _patch(monkeypatch, _script, [_FakeTool("web_search")])

    result = Research(_state())

    assert result["research_messages"][0].content == "final note"


def test_research_early_exit_after_search_and_two_fetches(monkeypatch):
    search = _FakeTool("web_search")
    fetch = _FakeTool("fetch_page")
    state = {"calls": 0}

    def _script(messages):
        if "No more tool calls" in str(messages[-1].content):
            return AIMessage(content="final note")
        state["calls"] += 1
        if state["calls"] == 1:
            return _tool_call("web_search", {"query": "q"}, call_id="s")
        return _tool_call(
            "fetch_page", {"url": "https://e.com", "snippet": "s"}, call_id="f"
        )

    _patch(monkeypatch, _script, [search, fetch])

    result = Research(_state())

    assert len(search.calls) == 1
    assert len(fetch.calls) == 2
    assert result["research_messages"][0].content == "final note"


def test_final_note_strips_accidental_tool_calls(monkeypatch):
    def _script(_messages):
        return AIMessage(
            content="note",
            tool_calls=[{"name": "web_search", "args": {}, "id": "x"}],
        )

    _patch(monkeypatch, _script, [_FakeTool("web_search")])

    result = Research(_state())
    message = result["research_messages"][0]

    assert message.content == "note"
    assert message.tool_calls == []


def test_research_propagates_turn_budget_exceeded(monkeypatch):
    def _script(_messages):
        raise TurnBudgetExceeded("calls", limit=1, observed=2)

    _patch(monkeypatch, _script, [_FakeTool("web_search")])

    try:
        Research(_state())
    except TurnBudgetExceeded:
        pass
    else:
        raise AssertionError("TurnBudgetExceeded must not be swallowed")


def test_research_max_iterations_reaches_final_note(monkeypatch):
    search = _FakeTool("web_search")
    fetch = _FakeTool("fetch_page")
    # Always ask for a fetch that is *not* counted past the cap logic by
    # alternating an unknown tool so no early-exit triggers.
    def _script(messages):
        if "No more tool calls" in str(messages[-1].content):
            return AIMessage(content="final note")
        return _tool_call("bogus")

    model = _patch(monkeypatch, _script, [search, fetch])

    result = Research(_state())

    # One iteration per MAX_ITERATIONS plus the forced final note.
    assert model.calls == MAX_ITERATIONS + 1
    assert result["research_messages"][0].content == "final note"
