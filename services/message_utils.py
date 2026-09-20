from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

CLARIFICATION_ANSWER_FLAG = "is_clarification_answer"

# Clear compensation/credential phrasings only. Ambiguous terms that caused
# false positives ("earnings", "bonus", "credential(s)") are deliberately
# excluded; the Postgres role remains the security boundary.
SENSITIVE_PATTERNS = (
    # Compensation / payroll
    "salary",
    "salaries",
    "compensation",
    "wage",
    "wages",
    "remuneration",
    "payroll",
    "paycheck",
    "get paid",
    "base pay",
    "annual pay",
    "pay rate",
    "income",
    "pay grade",
    "take-home pay",
    "net pay",
    "gross pay",
    # Credentials
    "password",
    "password hash",
    "api key",
    "access token",
    "client secret",
    "secret key",
)

_SENSITIVE_RE = re.compile(
    r"\b(?:" + "|".join(SENSITIVE_PATTERNS) + r")\b",
    re.IGNORECASE,
)

# "How much does Alice Example make?" — a person's pay, without matching
# company revenue questions like "how much do we make in sales?".
_MAKE_MONEY_RE = re.compile(
    r"\bhow much\b[^.?!]{0,30}\b(?:does|do)\b"
    r"(?!\s+(?:we|they|i|you)\b)[^.?!]{0,30}\bmake\b",
    re.IGNORECASE,
)


def mentions_sensitive_data(text: str | None) -> bool:
    """True if text mentions data guarded by elevated permissions.

    This is a UX fast-path, not a security boundary. The actual boundary is
    which tools are bound by ``permission_level`` and the Postgres role grants;
    a missed synonym here cannot grant access to sensitive tables.
    """
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return False

    match = _SENSITIVE_RE.search(normalized) or _MAKE_MONEY_RE.search(normalized)
    if match:
        logger.debug(
            "sensitive_intent_detected",
            extra={"matched": match.group(0).lower()},
        )
        return True
    return False

def content_to_text(content) -> str:
    """Normalise str | list[str|dict] (multimodal) content to a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return " ".join(parts)
    return str(content)


def is_clarification_answer(message) -> bool:
    extra = getattr(message, "additional_kwargs", None) or {}
    return bool(extra.get(CLARIFICATION_ANSWER_FLAG))


def clarification_answers(messages) -> list[str]:
    """Tagged clarification answers for the current turn only: those after
    the most recent untagged user request."""
    answers: list[str] = []
    for message in reversed(messages or []):
        if not isinstance(message, HumanMessage):
            continue
        if is_clarification_answer(message):
            text = " ".join(content_to_text(message.content).split())
            if text:
                answers.append(text)
            continue
        break
    return list(reversed(answers))


def latest_user_request(messages) -> Optional[str]:
   
    fallback: Optional[str] = None
    for message in reversed(messages or []):
        if not isinstance(message, HumanMessage):
            continue
        text = " ".join(content_to_text(message.content).split())
        if is_clarification_answer(message):
            if fallback is None:
                fallback = text
            continue
        return text
    return fallback