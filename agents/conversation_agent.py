from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from state.state import SupervisorState
from services.llm import llm
 
 
CONVO_SYSTEM_PROMPT = """You are the conversational agent for a company intelligence assistant.
You handle requests that don't need a specialist tool: general questions, clarifications,
small talk, or anything answerable directly from context.
Respond naturally and concisely based on the summary you're given."""

def Convo(state: SupervisorState) -> dict:
    """
    Handles requests that don't need a specialist: the Supervisor has
    already determined it can answer directly and pre-summarized the
    relevant context into current_task.
    """
    if state.current_task is None:
        raise RuntimeError(
            "Convo node reached with current_task=None. The Supervisor "
            "should always set current_task before routing here."
        )
    
    model = llm()

    msg = [SystemMessage(content=CONVO_SYSTEM_PROMPT)] + [HumanMessage(content=state.current_task)]

    response = model.invoke(msg)

    return {"messages": [response]}