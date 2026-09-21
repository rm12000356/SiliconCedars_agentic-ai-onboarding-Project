from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

CLARIFICATION_ANSWER_FLAG = "is_clarification_answer"

# Unambiguous compensation/credential terms only. Ambiguous synonyms
# ("compensation", "bonus", "income", "access token", "how much does X make")
# are deliberately excluded; the Postgres role remains the security boundary,
# and Finalize maps a real denial to a user-facing permission message.
SENSITIVE_PATTERNS = (
    "salary",
    "salaries",
    "password",
    "password hash",
    "api key",
)

_SENSITIVE_RE = re.compile(
    r"\b(?:" + "|".join(SENSITIVE_PATTERNS) + r")\b",
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

    match = _SENSITIVE_RE.search(normalized)
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