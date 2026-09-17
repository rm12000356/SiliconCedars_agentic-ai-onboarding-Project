import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from state.state import FactExtraction
from state.state import SupervisorState
from services.llm import llm
from services.memory import write_fact

logger = logging.getLogger(__name__)

IDENTITY_HINTS = [
    "my name is", "i'm ", "i am ", "call me", "i prefer", "i like",
    "please always", "in the future", "i work in", "i'm from",
]

NO_ANSWER_FALLBACK = (
    "I wasn't able to put together a response for that -- "
    "could you rephrase or add a bit more detail?"
)


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

    if state.last_result is not None:
        result = state.last_result
        if result.status == "done":
            content = result.summary
        else:
            content = f"{result.summary} ({result.issue or 'incomplete'})"

        if content and content.strip() and not _already_delivered(state.messages, content):
            update["messages"] = [AIMessage(content=content)]

        update["last_result"] = None
        update["clarification_count"] = 0

    user_id = (config.get("configurable") or {}).get("user_id")
    latest_human = _latest_human_message(state.messages)

    if user_id and latest_human and _might_contain_memorable_info(latest_human):
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