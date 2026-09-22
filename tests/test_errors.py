"""Unit tests for services.errors.classify_llm_error — no LLM, no network."""

from __future__ import annotations

from pydantic import ValidationError

from services.errors import classify_llm_error, is_provider_outage


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


_AGGREGATE = (
    "All configured LLM models failed. Last error: RateLimitError. "
    "Check GROQ_API_KEY, OPENROUTER_API_KEY and model names."
)


def test_aggregate_api_key_trailer_alone_is_not_auth():
    # The wrapper message mentions API keys but wraps no provider error.
    assert classify_llm_error(RuntimeError(_AGGREGATE)) is None


def test_aggregate_is_classified_by_chained_429():
    try:
        try:
            raise _HttpError(429, "Too Many Requests")
        except _HttpError as cause:
            raise RuntimeError(_AGGREGATE) from cause
    except RuntimeError as exc:
        assert classify_llm_error(exc) == "transient"


def test_aggregate_is_classified_by_chained_401():
    try:
        try:
            raise _HttpError(401, "Unauthorized")
        except _HttpError as cause:
            raise RuntimeError(_AGGREGATE) from cause
    except RuntimeError as exc:
        assert classify_llm_error(exc) == "auth"


def test_aggregate_is_classified_by_chained_timeout():
    try:
        try:
            raise TimeoutError("the request timed out")
        except TimeoutError as cause:
            raise RuntimeError(_AGGREGATE) from cause
    except RuntimeError as exc:
        assert classify_llm_error(exc) == "transient"


def test_is_provider_outage_true_for_auth_and_transient():
    assert is_provider_outage(_HttpError(401, "Unauthorized")) is True
    assert is_provider_outage(_HttpError(503, "Service Unavailable")) is True


def test_is_provider_outage_false_for_internal_bugs():
    assert is_provider_outage(RuntimeError("Missing runtime configuration")) is False
    assert is_provider_outage(ValidationError.from_exception_data("X", [])) is False
    assert is_provider_outage(ValueError("bad sql")) is False