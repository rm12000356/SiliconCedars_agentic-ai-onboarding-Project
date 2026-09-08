from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage
from state.state import SupervisorState
from services.llm import llm

MESSAGE_THRESHOLD = 12
KEEP_RECENT_MESSAGES = 6
TASK_HISTORY_KEEP_TURNS = 3


def memory_manager(state: SupervisorState) -> dict:
    update: dict = {}

    new_turn = state.turn_count + 1
    update["turn_count"] = new_turn
    print(f"[MEMORY] starting turn {new_turn}")

    # --- Prune/summarize old messages ---
    if len(state.messages) > MESSAGE_THRESHOLD:
        to_summarize = state.messages[:-KEEP_RECENT_MESSAGES]
        transcript = "\n".join(f"{m.type}: {m.content}" for m in to_summarize)

        print(f"[MEMORY] summarizing {len(to_summarize)} old messages")

        summary_response = llm().invoke([
            HumanMessage(
                content=(
                    "Summarize this conversation concisely, preserving important "
                    "facts, names, decisions, and open questions:\n\n" + transcript
                )
            )
        ])

        removals = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
        summary_message = SystemMessage(
            content=f"[Summary of earlier conversation]: {summary_response.content}"
        )

        update["messages"] = removals + [summary_message]
        print(f"[MEMORY] summary: {str(summary_response.content)[:200]!r}")

    # --- Prune old task_history entries ---
    cutoff = new_turn - TASK_HISTORY_KEEP_TURNS
    kept_history = [r for r in state.task_history if r.turn > cutoff]
    if len(kept_history) != len(state.task_history):
        print(
            f"[MEMORY] pruned task_history from {len(state.task_history)} "
            f"to {len(kept_history)} records (cutoff turn={cutoff})"
        )
    update["task_history"] = kept_history

    return update