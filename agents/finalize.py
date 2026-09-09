from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from state.state import  FactExtraction 
from state.state import SupervisorState
from services.llm import llm
from services.memory import write_fact

IDENTITY_HINTS = [
    "my name is", "i'm ", "i am ", "call me", "i prefer", "i like",
    "please always", "in the future", "i work in", "i'm from",
]


def _might_contain_memorable_info(text: str) -> bool:
    lower = text.lower()
    return any(hint in lower for hint in IDENTITY_HINTS)


def _latest_human_message(messages) -> str | None:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return None


def Finalize(state: SupervisorState, config: RunnableConfig) -> dict:
    """
    """
    update: dict = {}

    if state.last_result is not None:
        result = state.last_result
        if result.status == "done":
            content = result.summary
        else:
            content = f"{result.summary} ({result.issue or 'incomplete'})"
        update["messages"] = [AIMessage(content=content)]
        update["last_result"] = None

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
                print(f"[MEMORY] wrote fact for user {user_id}: {fact.key}={fact.value}")
        except Exception as e:
            # Long-term memory is a nice-to-have, not a hard dependency.
            # A failed extraction should never break finishing the turn.
            print(f"[MEMORY] fact extraction failed, skipping: {e}")

    return update