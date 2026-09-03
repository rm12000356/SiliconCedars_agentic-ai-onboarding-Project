from typing import Optional
from pydantic import BaseModel, Field
from state.state import MainRoute  # reuse the same Literal, don't redefine it


class SupervisorDecision(BaseModel):
    """
    Structured output schema for a single Supervisor routing decision.
    This is the LLM call's validation boundary, not persisted state.
    map_to_state() unpacks this into SupervisorState.next / current_task.
    """
    next: MainRoute = Field(
        description="Exactly one route to send the request to next."
    )
    current_task: str = Field(
        description="Concise actionable task for the routed specialist. "
                    "For convo specifically, this is a pre-summary of relevant "
                    "conversation context instead of a task instruction."
    )