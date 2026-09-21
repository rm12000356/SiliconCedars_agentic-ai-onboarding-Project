from __future__ import annotations

from agents.supervisor import get_supervisor_decision, get_workflow_plan
from services.llm import _is_retryable


class _HttpError(Exception):
    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        super().__init__(message or str(status_code))


def test_model_specific_400_falls_through():
    # A malformed tool call from one model should let the chain try the next.
    assert _is_retryable(
        _HttpError(400, "Tool call validation failed: tool_use_failed")
    )
    assert _is_retryable(
        _HttpError(400, "failed_generation: attempted to call tool 'X'")
    )


def test_generic_400_is_not_retryable():
    assert not _is_retryable(_HttpError(400, "context length exceeded"))


def test_401_is_not_retryable():
    assert not _is_retryable(_HttpError(401, "Unauthorized"))


def test_429_is_retryable():
    assert _is_retryable(_HttpError(429, "Too Many Requests"))


class _FakeStructured:
    def __init__(self, method, sink):
        self._method = method
        self._sink = sink

    def invoke(self, _prompt):
        self._sink.append(self._method)
        raise RuntimeError("schema failure")


class _FakeModel:
    def __init__(self, sink):
        self._sink = sink

    def with_structured_output(self, _schema, method=None, **_kwargs):
        return _FakeStructured(method, self._sink)


def _decision_context() -> dict:
    return {
        "messages": [],
        "last_result": None,
        "task_history": [],
        "known_facts": "",
        "conversation_summary": None,
    }


def test_supervisor_decision_never_uses_json_mode():
    sink: list = []
    get_supervisor_decision(_decision_context(), _FakeModel(sink))
    assert "function_calling" in sink
    assert "json_mode" not in sink


def test_workflow_plan_never_uses_json_mode():
    sink: list = []
    get_workflow_plan(_decision_context(), _FakeModel(sink))
    assert "function_calling" in sink
    assert "json_mode" not in sink
