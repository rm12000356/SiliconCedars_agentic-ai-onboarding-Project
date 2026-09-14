"""Unit tests for services.errors.classify_llm_error — no LLM, no network."""

from __future__ import annotations

from pydantic import ValidationError

from services.errors import classify_llm_error


class _HttpError(Exception):
    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        super().__init__(message or str(status_code))


def test_status_401_is_auth():
    assert classify_llm_error(_HttpError(401, "Unauthorized")) == "auth"


def test_status_403_is_auth():
    assert classify_llm_error(_HttpError(403, "Forbidden")) == "auth"


def test_status_429_is_transient():
    assert classify_llm_error(_HttpError(429, "Too Many Requests")) == "transient"


def test_status_503_is_transient():
    assert classify_llm_error(_HttpError(503, "Service Unavailable")) == "transient"


def test_timeout_message_without_status_is_transient():
    assert classify_llm_error(TimeoutError("the request timed out")) == "transient"


def test_internal_server_error_message_without_status_is_transient():
    assert classify_llm_error(RuntimeError("Internal Server Error")) == "transient"


def test_validation_error_is_schema_not_infra():
    try:
        raise ValidationError.from_exception_data("SupervisorDecision", [])
    except ValidationError as exc:
        assert classify_llm_error(exc) is None


def test_tool_use_failed_is_schema_not_infra():
    err = RuntimeError(
        "Tool choice is required, but model did not call a tool: tool_use_failed"
    )
    assert classify_llm_error(err) is None


def test_bare_digits_in_unrelated_message_are_not_transient():
    assert classify_llm_error(RuntimeError("see line 429 in parser")) is None