import math
from typing import Annotated, List, Optional
from typing_extensions import Literal
from pydantic import BaseModel, Field, field_validator
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

PlanRoute = Literal[
    "rag",
    "convo",
    "sql",
    "research",
    "visu",
    "clarification",
]

SubRoute = Literal[
    "researcher",
    "report",
    "end",
]


class SpecialistResult(BaseModel):
    """What a specialist hands back to the Supervisor: enough to decide the
    next hop, without raw tool metadata, similarity scores, or query plans."""
    source: SpecialistRoute = Field(description="Which specialist produced this result")
    summary: str = Field(description="The actual answer/content, already synthesized")
    status: Literal["done", "partial", "failed"] = Field(
        description="Whether the specialist fully completed its objective"
    )
    structured_data: Optional[list[dict]] = Field(
        default=None,
        description="Exact label/value pairs when the result is chartable, so "
                    "visu can consume real numbers instead of the LLM retyping them."
    )
    issue: Optional[str] = Field(
        default=None,
        description="Short reason if status is partial/failed, e.g. 'missing credentials for X'"
    )

class TaskRecord(BaseModel):
    """One specialist invocation during a user turn. Records the concrete task,
    not just the route, because the same specialist can run twice per turn."""

    turn: int = Field(
        description="User-turn number (from the persisted turn counter)."
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



class PlanItem(BaseModel):
    """One step of the up-front workflow plan.

    The supervisor's LLM decomposes the request into steps once per turn;
    execution then advances deterministically and writes each step's result
    back here, so Finalize can combine every completed result.
    """

    route: PlanRoute = Field(description="Specialist that runs this step")
    task: str = Field(description="Concrete task for that specialist")
    status: Literal["pending", "done", "failed", "skipped"] = Field(
        default="pending",
        description="Execution status of this plan step.",
    )
    result_summary: Optional[str] = Field(
        default=None, description="The step's synthesized result, if it ran."
    )
    issue: Optional[str] = Field(
        default=None, description="Structured issue code if the step failed."
    )
    structured_data: Optional[list[dict]] = Field(
        default=None,
        description="Chartable label/value rows, when the step produced them.",
    )
    data_source: Optional[Literal["database", "inline"]] = Field(
        default=None,
        description="For visu steps: 'database' charts structured_data from an "
                    "earlier sql step; 'inline' charts values supplied by the "
                    "user. None for non-visu steps.",
    )


class SupervisorState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = Field(
        description="Full conversation history; specialists get a filtered slice "
                    "via current_task, not this directly."
    )
    next: Optional[MainRoute] = Field(
        default=None,
        description="Single next hop, re-decided by the Supervisor each time it re-enters."
    )
    current_task: Optional[str] = Field(
        default=None,
        description="What the routed specialist actually needs to do, so it "
                    "doesn't have to parse full message history itself."
    )
    last_result: Optional[SpecialistResult] = Field(
        default=None,
        description="Most recent specialist result; what the Supervisor reacts "
                    "to when re-deciding."
    )
    task_history: List[TaskRecord] = Field(
        default_factory=list,
        description="Specialist invocations so far: the concrete task and its outcome."
    )
    plan: List[PlanItem] = Field(
        default_factory=list,
        description="Up-front workflow plan for the turn; each item accumulates "
                    "its own result. Reset by memory_manager."
    )
    plan_ready: bool = Field(
        default=False,
        description="True once the planner has produced this turn's plan; "
                    "cleared on clarification resume and each new turn."
    )
    turn_count: int = Field(
        default=0,
        description="Incremented once per new user turn by memory_manager "
                    "(not on interrupt resumes)."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to put to the human. Generated by the Supervisor, "
                    "not the node, because LangGraph replays a node on resume."
    )
    clarification_count: int = Field(
        default=0,
        description="Clarifications already asked this turn; reset by memory_manager."
    )
    conversation_summary: Optional[str] = Field(
        default=None,
        description="Rolling summary of pruned older messages; injected at the "
                    "top of the Supervisor prompt."
    )
    chart_path: Optional[str] = Field(
        default=None,
        description="Filesystem path of the most recent chart PNG; set by "
                    "Visualization and reset by memory_manager each turn."
    )
    outage: bool = Field(
        default=False,
        description="Set when the turn ended because no LLM provider was "
                    "usable, so Finalize emits a static message without calling "
                    "the model again."
    )
    hops: int = Field(
        default=0,
        description="Routed steps taken this turn; reset by memory_manager and "
                    "capped by MAX_HOPS as a loop backstop."
    )
    plan_note: Optional[str] = Field(
        default=None,
        description="User-facing note when the planner produced more asks than "
                    "MAX_PLAN_STEPS and the rest were dropped."
    )
    turn_cut_short: bool = Field(
        default=False,
        description="Set when the hop cap ended the turn early; Finalize says so."
    )


class SubGraphSupervisorState(BaseModel):
    """Private state for the Research subgraph; only some keys cross into the
    main graph."""

    messages: Annotated[List[AnyMessage], add_messages] = Field(
        description="Global message thread (not the full outer conversation)."
    )
    research_messages: Annotated[List[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="Messages scoped to the current research task."
    )
    next: Optional[SubRoute] = Field(
        default=None,
        description="Single next hop within the research subgraph."
    )
    task: str = Field(
        description="Research task handed down from the Supervisor's current_task."
    )
    research_attempts: int = Field(
        default=0,
        description="Deterministic count of Research invocations for this task."
    )
    research_succeeded: Optional[bool] = Field(
        default=None,
        description="Set by report_writer as a structural signal, not inferred from tone."
    )
    report_written: bool = Field(
        default=False,
        description="Set by report_writer once a report exists; sub_controller checks it "
                    "first and ends, preventing the report/controller infinite loop."
    )

class SubDecision(BaseModel):
    next: Literal["researcher", "report", "end"]
    reason: str = Field(description="Short explanation of the decision")

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

class ExtractedFact(BaseModel):
    key: str = Field(
        description="Short snake_case key, e.g. 'name', 'department', "
                    "'preference_chart_type'."
    )
    value: str = Field(description="The fact's value.")

class FactExtraction(BaseModel):
    facts: list[ExtractedFact] = Field(
        default_factory=list,
        description="Durable facts worth remembering across conversations: "
                    "name, department/role, stated preferences. Empty list "
                    "if nothing new or memorable was said. Do not extract "
                    "task-specific or one-off details, only standing facts.",
    )
