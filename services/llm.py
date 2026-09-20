from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from services.budget import get_budget

load_dotenv()

logger = logging.getLogger(__name__)

PRIMARY_MODEL = "openai/gpt-oss-120b"

FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "openai/gpt-oss-safeguard-20b",
]

OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 422}


RETRYABLE_TEXT_MARKERS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "service unavailable",
    "decommissioned",
    "model_not_found",
)

def _resolve_status(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    # Some client libraries nest the real HTTP status on a `.response`.
    response = getattr(exc, "response", None)
    nested_status = getattr(response, "status_code", None)
    if isinstance(nested_status, int):
        return nested_status

    return None


def _is_retryable(exc: Exception) -> bool:
    status = _resolve_status(exc)

    # An explicit status is authoritative in both directions.
    if status is not None:
        if status in RETRYABLE_STATUS_CODES:
            return True
        if status in NON_RETRYABLE_STATUS_CODES:
            return False
        # Unknown status (e.g. 404 decommissioned): fall through to the text
        # markers below.

    text = str(exc).lower()
    return any(marker in text for marker in RETRYABLE_TEXT_MARKERS)

class ResilientLLM:

    def __init__(
        self,
        candidates: list[tuple[str, Callable[[], Any]]],
        transforms: list[Callable[[Any], Any]] | None = None,
    ):
        self._candidates = candidates
        self._transforms = transforms or []

    def _with_transform(self, fn: Callable[[Any], Any]) -> "ResilientLLM":
        return ResilientLLM(self._candidates, self._transforms + [fn])

    def bind_tools(self, tools, **kwargs) -> "ResilientLLM":
        return self._with_transform(lambda client: client.bind_tools(tools, **kwargs))

    def with_structured_output(self, schema, **kwargs) -> "ResilientLLM":
        return self._with_transform(lambda client: client.with_structured_output(schema, **kwargs))

    def _build(self, factory: Callable[[], Any]) -> Any:
        client = factory()
        for transform in self._transforms:
            client = transform(client)
        return client

    def invoke(self, *args, **kwargs):
        budget = get_budget()
        last_exc: Exception | None = None

        for label, factory in self._candidates:
            if budget is not None:
                budget.check_and_count()

            try:
                client = self._build(factory)
            except Exception as e:
                logger.warning(
                    "[LLM] %s unavailable while constructing client: %s: %s",
                    label, type(e).__name__, e,
                )
                last_exc = e
                continue

            started = time.monotonic()
            try:
                logger.info("[LLM] Trying %s", label)
                response = client.invoke(*args, **kwargs)
            except Exception as e:
                if budget is not None:
                    budget.record_duration(time.monotonic() - started)
                last_exc = e
                if _is_retryable(e):
                    logger.warning(
                        "[LLM] %s failed (%s: %s) -> trying next model",
                        label, type(e).__name__, e,
                    )
                    continue
                logger.error(
                    "[LLM] %s failed with non-retryable error: %s: %s",
                    label, type(e).__name__, e,
                )
                raise

            if budget is not None:
                budget.record_duration(time.monotonic() - started)
                budget.record_usage(response)
            return response

        raise RuntimeError(
            "All configured LLM models failed. "
            f"Last error: {type(last_exc).__name__ if last_exc else '?'}: {last_exc}. "
            "Check GROQ_API_KEY, OPENROUTER_API_KEY and model names."
        )


def _make_groq(model_name: str) -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    return ChatGroq(model=model_name)


def _make_openrouter(model_name: str) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from .env")

    return ChatOpenAI(
        model=model_name,
        api_key=SecretStr(api_key),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/rm12000356/SiliconCedars_agentic-ai-onboarding-Project",
            "X-Title": "SiliconCedars",
        },
    )


def llm(model: str | None = None) -> ResilientLLM:

    if model is not None:
        if model.startswith("openrouter/") or (model.startswith("openai/") and "openrouter" in model):
            name = model.replace("openrouter/", "")
            return ResilientLLM([(f"openrouter:{name}", lambda n=name: _make_openrouter(n))])
        return ResilientLLM([(f"groq:{model}", lambda m=model: _make_groq(m))])
    
    candidates: list[tuple[str, Callable[[], Any]]] = [
        (f"groq:{m}", (lambda m=m: _make_groq(m))) for m in [PRIMARY_MODEL] + FALLBACK_MODELS
    ]
    candidates.append(
        (f"openrouter:{OPENROUTER_MODEL}", lambda: _make_openrouter(OPENROUTER_MODEL))
    )

    return ResilientLLM(candidates)