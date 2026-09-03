from typing import Annotated, List, Optional
from typing_extensions import Literal
from pydantic import BaseModel, Field
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

specialisationroute = Literal[
    "rag",
    "convo",
    "sql",
    "research",
    "visu"]

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
    source: specialisationroute = Field(description="Which specialist produced this result")
    summary: str = Field(description="The actual answer/content, already synthesized")
    status: Literal["done", "partial", "failed"] = Field(
        description="Whether the specialist fully completed its objective"
    )
    issue: Optional[str] = Field(
        default=None,
        description="Short reason if status is partial/failed, e.g. 'missing credentials for X'"
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