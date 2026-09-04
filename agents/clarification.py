from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage
from state.state import SupervisorState


def Clarification(state: SupervisorState) -> dict:
    """
    Pauses the graph and surfaces the Supervisor's clarifying question
    to the human. On resume, the human's answer is appended to messages
    so the Supervisor can re-decide with full context on its next turn.
    """
    question = state.current_task  # Supervisor already composed this

    answer = interrupt({"question": question})

    return {
        "messages": [HumanMessage(content=answer)],
        "current_task": None,   # consumed, Supervisor will set a fresh one
        "last_result": None,    # nothing to carry forward from a pause
    }

def resume_clarification(graph, thread_id: str, answer: str):
    """
    Called by the application layer once the human has answered the
    clarifying question. Resumes the paused graph at the interrupt point.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(Command(resume=answer), config=config)