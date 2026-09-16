from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage

CLARIFICATION_ANSWER_FLAG = "is_clarification_answer"

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