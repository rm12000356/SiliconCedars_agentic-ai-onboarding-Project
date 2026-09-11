from __future__ import annotations

_TRANSIENT_TOKENS = (
    "rate limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection",
    "connecterror",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "temporarily unavailable",
    "ssl",
    "internal server error",
    "server error",
)

_AUTH_TOKENS = (
    "invalid api key",
    "incorrect api key",
    "api key",
    "api_key",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid_api_key",
    "authenticationerror",
)


def classify_llm_error(exc: Exception) -> str | None:
    """
    'auth' | 'transient' | None (schema / unknown).

    Prefer HTTP status on the exception when present so we don't
    substring-match '429' inside unrelated messages.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status in (401, 403):
        return "auth"
    if status in (408, 429, 500, 502, 503, 504):
        return "transient"

    haystack = f"{type(exc).__name__} {exc}".lower()
    if any(t in haystack for t in _AUTH_TOKENS):
        return "auth"
    if any(t in haystack for t in _TRANSIENT_TOKENS):
        return "transient"
    return None