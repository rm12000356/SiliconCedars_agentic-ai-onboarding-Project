from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, SystemMessage
from state.state import SupervisorState
from langchain_core.runnables import RunnableConfig
from services.llm import llm
from state.structure_output import ClarificationOutput



def Clarification(state: SupervisorState) -> dict:
    """
    Pauses the graph and surfaces the Supervisor's clarifying question
    to the human. On resume, the human's answer is appended to messages
    so the Supervisor can re-decide with full context on its next turn.
    """
    model = llm()
    question_llm =model.with_structured_output(ClarificationOutput)
    question = state.current_task
    message = [SystemMessage(content=("Generate one concise clarification question that will help resolve the user's ambiguous request."))] + [HumanMessage(content=question)]

    result = question_llm.invoke(message)

    answer = interrupt({"question": result.question})

    return {
        "messages": [SystemMessage(content=state.current_task)] +[HumanMessage(content=answer)],
        "current_task": None,   # consumed, Supervisor will set a fresh one
        "last_result": None,    # nothing to carry forward from a pause
    }

def resume_clarification(graph, thread_id: str, answer: str, config: RunnableConfig):
    """
    Called by the application layer once the human has answered the
    clarifying question. Resumes the paused graph at the interrupt point.
    """
    return graph.invoke(Command(resume=answer), config=config)