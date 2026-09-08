import math
from typing import Annotated, List, Optional
from typing_extensions import Literal
from pydantic import BaseModel, Field , field_validator
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage



MainRoute = Literal[
    "rag",
    "convo",
    "sql",
    "research",       
    "visu",
    "clarification",
    "end",
]

SpecialistRoute = Literal[
    "rag",
    "convo",
    "sql",
    "research",
    "visu"
]

SubRoute = Literal[
    "researcher",
    "report",
    "end",
]


class SpecialistResult(BaseModel):
    """
    What a specialist hands back to the Supervisor.
    Deliberately thin: no raw tool metadata, no similarity scores,
    no query plans. Just enough for the Supervisor to decide what's next.
    """
    source: SpecialistRoute = Field(description="Which specialist produced this result")
    summary: str = Field(description="The actual answer/content, already synthesized")
    status: Literal["done", "partial", "failed"] = Field(
        description="Whether the specialist fully completed its objective"
    )
    structured_data: Optional[list[dict]] = Field(
        default=None,
        description="Exact tabular data (label/value pairs) when the result is "
                    "inherently chartable, e.g. SQL rows from a GROUP BY query. "
                    "Kept separate from summary deliberately: summary stays "
                    "thin, human-readable text; this field exists specifically "
                    "so a downstream specialist (visu) can consume exact "
                    "numbers without the LLM retyping them through prose, "
                    "which was unreliable and needlessly expensive."
    )
    issue: Optional[str] = Field(
        default=None,
        description="Short reason if status is partial/failed, e.g. 'missing credentials for X'"
    )

class TaskRecord(BaseModel):
    """
    One specialist invocation performed during a user turn.

    The same specialist may legitimately appear multiple times if it was
    given different tasks. History therefore records the concrete task,
    not merely the route.
    """

    turn: int = Field(
        description=(
            "User-turn number. Derived from the number of HumanMessages "
            "in the conversation."
        )
    )

    route: SpecialistRoute = Field(
        description="Specialist that executed this task"
    )

    task: str = Field(
        description="The concrete task assigned to the specialist"
    )

    status: Literal["done", "partial", "failed"] = Field(
        description="Outcome of this specialist invocation"
    )

    result_summary: str = Field(
        description="What the specialist produced or why it failed"
    )

    issue: Optional[str] = Field(
        default=None,
        description="Structured issue code/reason if applicable"
    )



class SupervisorState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = Field(
        description="Full conversation history. Supervisor reads all of it; "
                    "specialists get a filtered slice via current_task, not this directly."
    )
    next: Optional[MainRoute] = Field(
        default=None,
        description="Single next hop, re-decided by the Supervisor every time it's re-entered."
    )
    current_task: Optional[str] = Field(
        default=None,
        description="Supervisor's extraction of what the routed specialist actually needs "
                    "to do, so the specialist doesn't have to parse full message history itself."
    )
    last_result: Optional[SpecialistResult] = Field(
        default=None,
        description="What the most recently executed specialist returned. This is what the "
                    "Supervisor reacts to when re-deciding (e.g. status=partial -> clarification)."
    )
    task_history: List[TaskRecord] = Field(
        default_factory=list,
        description=(
            "History of specialist invocations performed in this conversation. "
            "Tracks what task was done and what result came from it."
        )
    )
    turn_count: int = Field(
        default=0,
        description=(
            "turn counter, set exactly once per external graph.invoke() call by memory_manager "
            "(the graph's entry node)"
        )
    )


class SubGraphSupervisorState(BaseModel):
    """
    Private state for the Research subgraph. Only relevant keys cross the
    boundary into the main graph, not this whole schema.
    """
    messages: Annotated[List[AnyMessage], add_messages] = Field(
        description="Local message thread scoped to the research task, "
                    "not the full outer conversation."
    )
    next: Optional[SubRoute] = Field(
        default=None,
        description="Single next hop within the research subgraph."
    )
    task: str = Field(
        description="The research task handed down from the main Supervisor's current_task."
    )
    research_attempts: int = Field(
        default=0,
        description="Deterministic count of how many times Research has been "
                    "invoked for this task."
    )
    research_succeeded: Optional[bool] = Field(
        default=None,
        description="Set by Report_W as a structural signal, not inferred from "
                    "the report's tone."
    )


class ChartSpec(BaseModel):
    chart_type: Literal["bar", "line", "pie"] = Field(
        description="The kind of chart to draw."
    )
    title: str = Field(
        min_length=1,
        max_length=120,
        description="A concise chart title."
    )
    labels: list[str] = Field(
        description="Category labels, one per value."
    )
    values: list[float] = Field(
        description="Numeric values, one per label."
    )

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Chart values must be finite numbers.")
        return values

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: list[str]) -> list[str]:
        if any(not label.strip() for label in labels):
            raise ValueError("Chart labels cannot be empty.")
        return labels