from langchain_core.messages import AIMessage
from state.state import SupervisorState


def Finalize(state: SupervisorState) -> dict:
    """
    Runs once, right before the graph actually ends. Converts the last
    specialist's result (if any) into a user-facing AIMessage.

    Convo doesn't need this: it already appends its own AIMessage
    directly to messages, so last_result stays None on that path and
    this node has nothing to do. This only matters for sql/rag/visu/
    research, which produce a SpecialistResult but never touch
    messages themselves.
    """
    if state.last_result is None:
        return {}

    result = state.last_result

    if result.status == "done":
        content = result.summary
    else:
        content = f"{result.summary} ({result.issue or 'incomplete'})"

    return {
        "messages": [AIMessage(content=content)],
        "last_result": None, 
    }