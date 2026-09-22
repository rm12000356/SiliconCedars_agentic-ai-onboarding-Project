import logging
from typing import NamedTuple, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from state.state import FactExtraction
from state.state import SupervisorState
from services.llm import llm
from services.budget import get_budget
from services.memory import write_fact

logger = logging.getLogger(__name__)

# Explicit durable-memory intent only. Bare "i'm"/"i am" triggered an
# extraction call on almost every message, including untrusted content.
IDENTITY_HINTS = [
    "my name is", "call me", "remember that", "please remember",
    "i prefer", "i like", "please always", "in the future",
    "i work in", "i'm from", "i am from", "note that i",
]

NO_ANSWER_FALLBACK = (
    "I wasn't able to put together a response for that -- "
    "could you rephrase or add a bit more detail?"
)

OUTAGE_MESSAGE = (
    "The assistant is temporarily unavailable due to a service issue. "
    "Please try again in a moment."
)

INTERNAL_ERROR_MESSAGE = (
    "Something went wrong while handling that request. "
    "Please try again or rephrase it."
)

TURN_CUT_SHORT_NOTE = (
    "This turn was cut short because it needed more steps than the "
    "assistant allows in one go. Please narrow the request."
)

# User-facing text for known internal failure codes. Raw issue strings can
# contain database/provider errors and must never be shown to the user.
_GENERIC_FAILURE = "I wasn't able to complete that request."

_ISSUE_MESSAGES = {
    "permission_denied": "You don't have permission to access that data.",
    "invalid_request": (
        "I couldn't complete that database request. Please rephrase your question."
    ),
    "budget_exceeded": (
        "That request was too complex to finish in one go. Please narrow it down."
    ),
    "rag_unavailable": "Internal document search is temporarily unavailable.",
    "no_matching_documents": (
        "I couldn't find any internal documents matching that request."
    ),
    "research_no_results": (
        "I couldn't find reliable external information for that request."
    ),
    "research_empty_report": "The research step didn't produce a usable report.",
    "visualization_failed": "I couldn't generate that chart.",
    "invalid_chart_spec": "That request didn't contain valid chart data.",
    "no_data_for_chart": (
        "I couldn't build that chart because the data it needed wasn't available."
    ),
    "not_chartable": (
        "I couldn't build that chart because the result doesn't have a label "
        "and a number (a chart needs exactly two columns)."
    ),
    "turn_cut_short": "This turn needed more steps than allowed in one go.",
    "empty_answer": "I couldn't produce an answer from the data retrieved.",
    "empty_synthesis": "I couldn't produce an answer from the data retrieved.",
}


SYNTHESIS_PROMPT = (
    "You are the final responder for a company intelligence assistant. "
    "Combine the specialist results below into ONE clear, coherent answer to "
    "the user. Preserve every concrete fact (numbers, names, sources). Do not "
    "invent anything. If a result failed or is incomplete, briefly note the "
    "limitation. Do not mention specialists, steps, or internal tooling."
)


class _ResultView(NamedTuple):
    status: str
    summary: str
    issue: Optional[str]


def _issue_code(issue: str | None) -> str:
    return (issue or "").split(":", 1)[0].strip()


def _completed_results(state: SupervisorState) -> list[_ResultView]:
    """Completed plan results, or the legacy single last_result when no plan is
    present (keeps direct callers and older checkpoints working)."""
    results = [
        _ResultView(item.status, item.result_summary or "", item.issue)
        for item in state.plan
        if item.status in ("done", "failed", "skipped") and item.result_summary
    ]
    if not results and state.last_result is not None:
        lr = state.last_result
        results.append(_ResultView(lr.status, lr.summary, lr.issue))
    return results


def _render_single(result: _ResultView) -> str:
    if result.status == "done":
        return result.summary
    return _user_facing_failure(result)


def _synthesize_results(results: list[_ResultView]) -> str:
    joined = "\n\n".join(
        f"[{i + 1}] status={r.status}"
        + (f" issue={_issue_code(r.issue)}" if r.issue else "")
        + f"\n{r.summary}"
        for i, r in enumerate(results)
    )
    try:
        response = llm().invoke([
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(content=joined),
        ])
        text = str(response.content).strip()
        if text:
            return text
    except Exception as e:
        logger.warning(
            "final_synthesis_failed",
            extra={"error": f"{type(e).__name__}: {e}"},
        )

    # Deterministic fallback: stitch the summaries together.
    return "\n\n".join(r.summary for r in results if (r.summary or "").strip())


def _user_facing_failure(result) -> str:
    code = _issue_code(result.issue)
    if code in _ISSUE_MESSAGES:
        return _ISSUE_MESSAGES[code]

    summary = (result.summary or "").strip()
    if summary and (not result.issue or result.issue not in summary):
        return summary
    return _GENERIC_FAILURE


def _might_contain_memorable_info(text: str) -> bool:
    lower = text.lower()
    return any(hint in lower for hint in IDENTITY_HINTS)


def _latest_human_message(messages) -> str | None:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return None


def _last_is_ai_message(messages) -> bool:
    if not messages:
        return False

    last_message = messages[-1]

    return (
        isinstance(last_message, AIMessage)
        and not last_message.tool_calls
    )


def _turn_has_assistant_output(messages) -> bool:
    return _last_is_ai_message(messages) and bool(str(messages[-1].content).strip())


def _already_delivered(messages, content: str) -> bool:
    if not _last_is_ai_message(messages):
        return False
    return str(messages[-1].content).strip() == str(content).strip()


def Finalize(state: SupervisorState, config: RunnableConfig) -> dict:
    update: dict = {}

    if state.outage:
        # No provider was usable. Emit a static message and make no LLM calls.
        if not _already_delivered(state.messages, OUTAGE_MESSAGE):
            update["messages"] = [AIMessage(content=OUTAGE_MESSAGE)]
        update["last_result"] = None
        update["clarification_count"] = 0
        return update

    results = _completed_results(state)

    if len(results) == 1:
        content = _render_single(results[0])
    elif len(results) >= 2:
        content = _synthesize_results(results)
    else:
        content = None

    extras = [
        note
        for note in (
            state.plan_note,
            TURN_CUT_SHORT_NOTE if state.turn_cut_short else None,
        )
        if note
    ]
    if extras:
        prefix = f"{content}\n\n" if content and content.strip() else ""
        content = prefix + "\n\n".join(extras)

    if content and content.strip() and not _already_delivered(state.messages, content):
        update["messages"] = [AIMessage(content=content)]

    update["last_result"] = None
    update["clarification_count"] = 0

    user_id = (config.get("configurable") or {}).get("user_id")
    latest_human = _latest_human_message(state.messages)

    budget = get_budget()
    budget_spent = budget is not None and budget.exhausted()

    if (
        user_id
        and latest_human
        and _might_contain_memorable_info(latest_human)
        and not budget_spent
    ):
        try:
            extractor = llm().with_structured_output(FactExtraction)
            extraction = extractor.invoke([
                SystemMessage(
                    content="Extract any durable facts about the user worth "
                            "remembering across conversations: name, "
                            "department/role, stated preferences. Return an "
                            "empty list if nothing new or memorable is present."
                ),
                HumanMessage(content=latest_human),
            ])
            if not isinstance(extraction, FactExtraction):
                extraction = FactExtraction.model_validate(extraction)

            for fact in extraction.facts:
                write_fact(user_id, fact.key, fact.value)
                logger.info(
                    "[MEMORY] wrote fact for user %s: %s=%s",
                    user_id, fact.key, fact.value,
                )
        except Exception as e:
            logger.warning("[MEMORY] fact extraction failed, skipping: %s", e)

    if "messages" not in update and not _turn_has_assistant_output(state.messages):
        update["messages"] = [AIMessage(content=NO_ANSWER_FALLBACK)]

    return update