from langchain_core.messages import SystemMessage, HumanMessage
from state.state import SupervisorState, SpecialistResult
from services.llm import llm


CONVO_SYSTEM_PROMPT = """You are the conversational agent for a company intelligence assistant.
You handle requests that don't need a specialist tool: general questions, clarifications,
small talk, or anything answerable directly from context.
Respond naturally and concisely based on the summary you're given."""


def Convo(state: SupervisorState) -> dict:
    """
    Handles requests that don't need a specialist tool.

    Returns both:
    - the actual AI response as a message
    - a SpecialistResult so the Supervisor knows this task completed
    """

    if state.current_task is None:
        raise RuntimeError(
            "Convo node reached with current_task=None. The Supervisor "
            "should always set current_task before routing here."
        )

    print(f"[CONVO] current_task={state.current_task!r}")

    model = llm()

    msg = [
        SystemMessage(content=CONVO_SYSTEM_PROMPT),
        HumanMessage(content=state.current_task),
    ]

    response = model.invoke(msg)

    result = SpecialistResult(
        source="convo",
        summary=str(response.content),
        status="done",
    )

    return {
        "messages": [response],
        "last_result": result,
    }