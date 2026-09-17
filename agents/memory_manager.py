import logging

from langchain_core.messages import RemoveMessage, HumanMessage
from state.state import SupervisorState
from services.llm import llm

logger = logging.getLogger(__name__)

MESSAGE_THRESHOLD = 12
KEEP_RECENT_MESSAGES = 6
TASK_HISTORY_KEEP_TURNS = 3


def memory_manager(state: SupervisorState) -> dict:
    update: dict = {}

    new_turn = state.turn_count + 1
    update["turn_count"] = new_turn
    update["clarification_count"] = 0
    update["chart_path"] = None
    logger.info("[MEMORY] starting turn %s", new_turn)

    if len(state.messages) > MESSAGE_THRESHOLD:
        to_summarize = state.messages[:-KEEP_RECENT_MESSAGES]
        transcript = "\n".join(f"{m.type}: {m.content}" for m in to_summarize)

        if state.conversation_summary:
            transcript = (
                f"Summary of even earlier conversation: "
                f"{state.conversation_summary}\n\n{transcript}"
            )
        
        logger.debug("[MEMORY] summarizing %s old messages", len(to_summarize))

        summary_response = llm().invoke([
            HumanMessage(
                content=(
                    "Summarize this conversation concisely, preserving important "
                    "facts, names, decisions, and open questions:\n\n" + transcript
                )
            )
        ])

        removals = [RemoveMessage(id=m.id) for m in to_summarize if m.id]

        update["messages"] = removals
        update["conversation_summary"] = str(summary_response.content)
        logger.debug("[MEMORY] summary: %r", str(summary_response.content)[:200])

    cutoff = new_turn - TASK_HISTORY_KEEP_TURNS
    kept_history = [r for r in state.task_history if r.turn > cutoff]
    if len(kept_history) != len(state.task_history):
        logger.debug(
            "[MEMORY] pruned task_history from %s to %s records (cutoff turn=%s)",
            len(state.task_history), len(kept_history), cutoff,
        )
    update["task_history"] = kept_history

    return update