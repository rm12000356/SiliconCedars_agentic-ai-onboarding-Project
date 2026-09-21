from typing import Literal

from pydantic import BaseModel, Field
from state.state import MainRoute, PlanRoute


class PlanStep(BaseModel):
    route: PlanRoute = Field(
        description="Specialist that should run this step of the workflow."
    )
    task: str = Field(
        description="Concise, self-contained task for that specialist."
    )
    data_source: Literal["database", "inline"] | None = Field(
        default=None,
        description="For visu steps only: 'database' when the chart uses "
                    "structured_data produced by an earlier sql step; 'inline' "
                    "when the user's message itself supplies the chart values. "
                    "Null for all other routes.",
    )


class WorkflowPlan(BaseModel):
    """The LLM's up-front decomposition of the user's request into an ordered
    workflow. Execution is deterministic after this; the plan is not re-derived
    after each hop."""

    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Ordered steps. One step for a single-intent request; a "
                    "clarification step alone when the request is unclear.",
    )


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

class ClarificationOutput(BaseModel):
    question: str = Field(
        description="The single clarification question to ask the user."
    )

class ReportOutput(BaseModel):
    content: str = Field(description="The final report text to show the user.")
    success: bool = Field(
        description="True only if genuine information answering the task was found. "
                    "False if the research material shows repeated failures, blocked "
                    "sources, or no usable data, regardless of how the report is worded."
    )