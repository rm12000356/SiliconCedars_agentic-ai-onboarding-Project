import logging

from langchain_core.messages import SystemMessage
from state.state import SupervisorState, SpecialistResult
from services.llm import llm
from services.budget import TurnBudgetExceeded

logger = logging.getLogger(__name__)


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

    logger.debug("[CONVO] current_task=%r", state.current_task)

    model = llm()

    recent = state.messages[-6:]
    system = f"{CONVO_SYSTEM_PROMPT}\n\nTask: {state.current_task}"
    msg = [SystemMessage(content=system), *recent]

    try:
        response = model.invoke(msg)
    except TurnBudgetExceeded as e:
        logger.warning("[CONVO] turn budget exceeded: %s", e)
        return {
            "last_result": SpecialistResult(
                source="convo",
                summary=(
                    "This request could not be completed within the allowed "
                    "budget for a single turn."
                ),
                status="failed",
                issue="budget_exceeded",
            )
        }

    result = SpecialistResult(
        source="convo",
        summary=str(response.content),
        status="done",
    )

    return {
        "messages": [response],
        "last_result": result,
    }