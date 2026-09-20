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

# Precise phrases only. Bare "api key" / "api_key" used to match the
# aggregate "Check GROQ_API_KEY..." message and misclassify every outage as
# auth, including 429s and timeouts.
_AUTH_TOKENS = (
    "invalid api key",
    "incorrect api key",
    "invalid_api_key",
    "authentication",
    "unauthorized",
    "forbidden",
    "authenticationerror",
)


def _resolve_status(exc: Exception) -> int | None:
    """HTTP status from the exception or a nested `.response`, if present."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    http_status = getattr(exc, "http_status", None)
    if isinstance(http_status, int):
        return http_status
    response = getattr(exc, "response", None)
    nested = getattr(response, "status_code", None)
    if isinstance(nested, int):
        return nested
    return None


def _iter_causes(exc: Exception):
    """Walk the exception chain so an aggregate error is classified by the
    underlying provider failure rather than its own wrapper message."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _classify_single(exc: Exception) -> str | None:
    status = _resolve_status(exc)
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


def classify_llm_error(exc: Exception) -> str | None:
    """
    'auth' | 'transient' | None (schema / unknown).

    Prefer HTTP status on the exception when present so we don't
    substring-match '429' inside unrelated messages. Classifies the first
    exception in the cause chain that yields a verdict, so a wrapper
    RuntimeError is judged by the provider error it wraps.
    """
    for candidate in _iter_causes(exc):
        result = _classify_single(candidate)
        if result is not None:
            return result
    return None