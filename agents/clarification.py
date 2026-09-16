from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage
from state.state import SupervisorState
from langchain_core.runnables import RunnableConfig

from services.message_utils import CLARIFICATION_ANSWER_FLAG

DEFAULT_QUESTION = "Could you clarify what you'd like me to do?"

def Clarification(state: SupervisorState) -> dict:
    """
    Pauses the graph and surfaces the Supervisor's clarifying question
    to the human. On resume, the human's answer is appended to messages
    so the Supervisor can re-decide with full context on its next turn.
    """
    question = (state.clarification_question or "").strip() or DEFAULT_QUESTION

    answer = interrupt({"question": question})

    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "messages": [
            AIMessage(content=question),
            HumanMessage(
                content=answer,
                additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
            ),
        ],
        "current_task": None,            
        "clarification_question": None,  
        "clarification_count": state.clarification_count + 1,
        "last_result": None,             
    }

def resume_clarification(graph, thread_id: str, answer: str, config: RunnableConfig):
    """
    Called by the application layer once the human has answered the
    clarifying question. Resumes the paused graph at the interrupt point.
    """
    return graph.invoke(Command(resume=answer), config=config)