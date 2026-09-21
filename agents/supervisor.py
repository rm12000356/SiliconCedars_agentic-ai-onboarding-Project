from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from state.state import SupervisorState, TaskRecord, SpecialistResult, PlanItem
from state.structure_output import (
    SupervisorDecision,
    ClarificationOutput,
    WorkflowPlan,
)
from services.message_utils import (
    clarification_answers,
    latest_user_request,
    mentions_sensitive_data,
)
from services.llm import llm
from services.memory import format_facts_for_prompt
from services.errors import classify_llm_error, LLMOutageError
from services.budget import get_budget, TurnBudgetExceeded
from agents.clarification import DEFAULT_QUESTION

logger = logging.getLogger(__name__)

MAX_CLARIFICATIONS_PER_TURN = 1
MAX_PLAN_STEPS = 3
MAX_HOPS = MAX_PLAN_STEPS + MAX_CLARIFICATIONS_PER_TURN + 1


def _end_decision() -> SupervisorDecision:
    return SupervisorDecision(next="end", current_task="")

SUPERVISOR_SYSTEM_PROMPT = """You are the routing supervisor for a company intelligence assistant.

Decide the single best next route for the latest user request.

### Routes
- rag: Internal company documents, policies, procedures, lessons-learned, and organizational knowledge. Choose this when the answer lives in a document: what a policy says, what a procedure is, or what was learned from a past project. Do NOT use it for concrete records in the operational database.
- research: External/public information, including current or time-sensitive facts (today's weather, the latest price, who currently holds a position, recent news). Use it whenever the answer is not company-internal and may have changed since training, even if the user did not say "research". Never use it for company-internal topics.
- sql: Live structured data from the operational database — counts, sums, lists, filters, rankings, and lookups of specific records such as one person's department, salary, or employee ID, a sale's amount/date, or a credential row. Choose sql whenever the answer is a field value in a table, even when the question names a person and is phrased like "What is <name>'s <field>?" or "Find the <id> for <name>".
- visu: Build a chart. Choose it when structured_data already exists from a previous result, or when the user provides the chart values inline; the user wants a chart/graph/plot.
- convo: You can answer directly (greetings, goodbyes, definitions, small talk, clarifying your own previous answer, or light synthesis).
- clarification: The request itself is genuinely unclear — you do not understand what the user wants. Do NOT use this just because the request might need multiple steps.
- end: Nothing further needs to happen.

### Decision rules
- Chart requests over database data without existing structured_data → sql (this is normal, not ambiguity).
- Chart requests whose values are given inline (e.g. "pie chart: 60% EU, 25% MENA, 15% APAC") → visu; no sql is needed.
- Prefer rag over research for anything internal.
- Prefer convo over clarification whenever the request is understandable.
- Goodbyes, thanks, and small talk → convo.
- Facts about a specific record/entity in the database (a named person's department, salary, or ID; a sale row; a count or list of rows) → sql, even when phrased conversationally.
- Questions about what a document/policy/procedure says, or what was learned from past projects → rag.
- Questions about current, latest, or time-sensitive external facts (current CEO/president, latest price, today's weather, recent news) → research, even without an explicit "research" verb.
- Definitions and general knowledge (e.g. "What does SQL stand for?", "What does RAG mean?") → convo, never sql.
- Focus almost exclusively on the latest user message and the latest specialist result. Ignore older conversation history unless it is directly needed to understand the current request.

### sql vs rag
Decide by where the answer comes from:
- A field inside a database table (employee, department, salary, sales, credentials, row counts) → sql.
- A document, policy, procedure, or lessons-learned note → rag.

### Examples
- "How many employees are there?" → sql
- "Make a pie chart: 60% EU, 25% MENA, 15% APAC." → visu
- "What is Alice Example's department?" → sql
- "Find the employee ID for Rami Noueihed." → sql
- "What is Rami Noueihed's salary?" → sql
- "What is the company remote work policy?" → rag
- "What did we learn about SQL security from past internal projects?" → rag
- "What does SQL stand for?" → convo
- "Who is the current CEO of OpenAI?" → research
- "What is the latest price of gold?" → research
- "What is the latest sale_date in the sales table?" → sql (company-internal data, not external research)
- "How many employees are there, and what does the remote work policy say?" → sql first; once the count is answered, the remaining policy question → rag

### current_task
- convo → short pre-summary of relevant context
- all other routes → concise, actionable task description
- end → leave empty


"""

PLANNER_SYSTEM_PROMPT = """You are the planning supervisor for a company intelligence assistant.

Decompose the latest user request into an ordered workflow of specialist steps. You are called ONCE per request; execution is then deterministic, so the plan must be complete and correctly ordered.

### Available steps
- sql: live structured data from the operational database — counts, sums, lists, a named person's department/salary, a sale row, a credential row.
- rag: internal company documents, policies, procedures, lessons-learned.
- research: external/public information, including current or time-sensitive facts.
- visu: build a chart. Include it after a sql step when the user asks for a chart/graph/plot over database data, or as the ONLY step when the user provides the chart values inline. Every visu step MUST set data_source: "database" when it charts structured_data from an earlier sql step, or "inline" when the user's message supplies the values. For "inline", restate the exact labels and values in the task.
- convo: answer directly (greetings, definitions, small talk, light synthesis).
- clarification: the request is genuinely unclear. Use this as the ONLY step.

### Rules
- One step per distinct ask, in the order they should run.
- A single-intent request has exactly one step.
- A two-part request ("how many employees, and what does the policy say?") has two steps: sql then rag.
- A chart over database data is two steps: sql then visu.
- A chart whose values are given in the message (e.g. "pie chart: 60% EU, 25% MENA, 15% APAC") is a single visu step; no sql is needed.
- Prefer rag over research for company-internal topics.
- Current/latest external facts (current CEO, latest price, today's weather, recent news) → research.
- "What did we learn about ..." or "what does the policy say" → rag, even if the topic names a table.
- Do not include an "end" step.
- Keep each task concise, self-contained, and actionable.
- Maximum 3 steps.

### Examples
- "How many employees are there?" → [sql]
- "What is the company remote work policy?" → [rag]
- "Who is the current CEO of OpenAI?" → [research]
- "Hello" → [convo]
- "How many employees are there, and what does the remote work policy say?" → [sql, rag]
- "Show sales by region as a pie chart." → [sql, visu]
- "Make a pie chart: 60% EU, 25% MENA, 15% APAC." → [visu]
- "What did we learn about SQL security from past projects?" → [rag]
- "Tell me about the numbers." → [clarification]
"""

_OUTAGE_TASK = (
    "Explain that the assistant is temporarily unavailable due to a service "
    "issue, not because the request was unclear. Ask them to try again later. "
    "Do not invent an answer."
)

CLARIFICATION_QUESTION_PROMPT = (
    "Generate one concise clarification question that will help resolve the "
    "user's ambiguous request. Ask exactly one question."
)

def supervisor_agent(state: SupervisorState, config: RunnableConfig) -> dict:
    """Record the prior result, decide deterministically if possible, then
    apply post-guards and map the decision into a state update."""
    task_history = list(state.task_history)

    if state.last_result is not None and state.current_task:
        current_turn = get_current_turn(state)
        task_record = TaskRecord(
            turn=current_turn,
            route=state.last_result.source,
            task=state.current_task,
            status=state.last_result.status,
            result_summary=state.last_result.summary,
            issue=state.last_result.issue,
        )
        task_history.append(task_record)
        logger.debug(
            "recorded_task",
            extra={
                "turn": current_turn,
                "route": task_record.route,
                "task": task_record.task,
                "status": task_record.status,
            },
        )

    plan = [item.model_copy() for item in state.plan]
    outage = False

    # Write the step we just routed to (the first pending item) back into the
    # plan, then advance deterministically. No per-hop LLM routing.
    if state.plan_ready and state.last_result is not None:
        plan = _complete_current_step(plan, state.last_result)

    # No budget left to plan anything: close the turn. If a plan already
    # exists it still advances, so a deterministic visu step can render.
    budget = get_budget()
    if budget is not None and budget.exhausted() and not state.plan_ready:
        logger.warning("turn_budget_exhausted_forcing_end")
        update = map_to_state(_end_decision())
        update["task_history"] = task_history
        update["plan"] = plan
        update["plan_ready"] = True
        update["hops"] = state.hops
        update["plan_note"] = state.plan_note
        update["turn_cut_short"] = state.turn_cut_short
        return update

    user_id = (config.get("configurable") or {}).get("user_id")
    plan_note = state.plan_note

    if not state.plan_ready:
        try:
            context = gather_context(state, task_history, user_id)
            plan = get_workflow_plan(context, llm())
        except LLMOutageError as e:
            logger.warning("supervisor_outage_ending_turn", extra={"kind": str(e)})
            outage = True
            plan = []
        except TurnBudgetExceeded as e:
            logger.warning("supervisor_turn_budget_exceeded", extra={"error": str(e)})
            plan = []
        except Exception as e:
            logger.warning(
                "workflow_planning_failed_falling_back",
                extra={"error": f"{type(e).__name__}: {e}"},
            )
            plan = []

        if not plan and not outage:
            # Fallback: one single-decision LLM call, stored as a one-step plan.
            fallback: Optional[SupervisorDecision] = None
            try:
                context = gather_context(state, task_history, user_id)
                fallback = get_supervisor_decision(context, llm())
            except LLMOutageError as e:
                logger.warning("supervisor_outage_ending_turn", extra={"kind": str(e)})
                outage = True
            except TurnBudgetExceeded as e:
                logger.warning("supervisor_turn_budget_exceeded", extra={"error": str(e)})

            if fallback is not None and fallback.next != "end":
                plan = [
                    PlanItem(
                        route=fallback.next,
                        task=fallback.current_task,
                        data_source="inline" if fallback.next == "visu" else None,
                    )
                ]
            elif fallback is None and not outage:
                request = latest_user_request(state.messages) or ""
                if mentions_sensitive_data(request):
                    plan = [PlanItem(route="sql", task=request)]
                else:
                    plan = [PlanItem(route="convo", task=_OUTAGE_TASK)]

        if not outage and plan:
            plan, plan_note = _validate_plan(plan, state)

        plan = _apply_clarification_cap(plan, state)

    plan = _maybe_insert_visu(plan, state)
    plan = _skip_blocked_steps(plan)
    plan = _skip_unbudgeted_visu(plan, state)

    decision = _end_decision() if outage else _next_plan_decision(plan)

    # Cheap loop backstop: end the turn when it exceeds the legitimate step
    # budget. This should be unreachable; if it fires, a plan invariant broke.
    hops = state.hops
    turn_cut_short = state.turn_cut_short
    if decision.next != "end":
        hops += 1
        if hops > MAX_HOPS:
            logger.error(
                "hop_cap_exceeded_cutting_turn_short",
                extra={
                    "hops": hops,
                    "plan": [(item.route, item.status) for item in plan],
                },
            )
            turn_cut_short = True
            for item in plan:
                if item.status == "pending":
                    item.status = "failed"
                    item.issue = "turn_cut_short"
                    item.result_summary = None
            decision = _end_decision()

    logger.debug(
        "final_decision",
        extra={
            "next": decision.next,
            "current_task": decision.current_task,
            "plan": [(item.route, item.status) for item in plan],
        },
    )

    update = map_to_state(decision)
    update["task_history"] = task_history
    update["plan"] = plan
    update["plan_ready"] = True
    update["hops"] = hops
    update["plan_note"] = plan_note
    update["turn_cut_short"] = turn_cut_short

    next_item = _next_pending(plan)
    if decision.next == "visu" and next_item is not None and next_item.route == "visu":
        if next_item.data_source == "inline":
            # Inline charts must extract their values from the current task; do
            # not let a stale structured_data result override the user's numbers.
            update["last_result"] = None
        else:
            index = next(
                (i for i, item in enumerate(plan) if item is next_item), None
            )
            rows = _rows_upstream(plan, index) if index is not None else None
            if rows:
                # Re-supply the upstream sql rows explicitly: an intervening
                # step (e.g. rag) may have cleared last_result.
                update["last_result"] = SpecialistResult(
                    source="sql",
                    summary="Upstream structured data for the chart.",
                    status="done",
                    structured_data=rows,
                )

    if decision.next == "clarification":
        update["clarification_question"] = _generate_clarification_question(decision)

    if outage:
        update["outage"] = True

    return update


def _generate_clarification_question(decision: SupervisorDecision) -> str:
    """Best-effort. A failure here must not break the turn."""
    try:
        model = llm().with_structured_output(ClarificationOutput)
        result = model.invoke([
            SystemMessage(content=CLARIFICATION_QUESTION_PROMPT),
            HumanMessage(
                content=decision.current_task or "The user's request was unclear."
            ),
        ])
        if not isinstance(result, ClarificationOutput):
            result = ClarificationOutput.model_validate(result)
        question = (result.question or "").strip()
        if question:
            return question
    # Best-effort: any generation failure (provider, schema, budget) falls
    # back to the canned question rather than breaking the turn.
    except Exception as e:
        logger.warning(
            "clarification_question_generation_failed",
            extra={"error": f"{type(e).__name__}: {e}"},
        )
    return DEFAULT_QUESTION


def build_planning_prompt(context: dict) -> list:
    messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT)]

    if context.get("conversation_summary"):
        messages.append(
            SystemMessage(
                content=f"[Summary of earlier conversation]: "
                        f"{context['conversation_summary']}"
            )
        )

    if context.get("known_facts"):
        messages.append(SystemMessage(content=context["known_facts"]))

    messages.extend(context["messages"][-6:])
    return messages


def _plan_from_workflow(workflow: WorkflowPlan) -> list[PlanItem]:
    """Map the planner's workflow into plan items; filtering, ordering
    guarantees, and the step cap live in _validate_plan."""
    plan: list[PlanItem] = []
    for index, step in enumerate(workflow.steps):
        if not step.task or not step.task.strip():
            continue

        data_source = step.data_source
        if step.route == "visu" and data_source is None:
            # Deterministic fallback when the planner omits the field: a visu
            # that follows a sql step charts that data; a standalone visu
            # charts inline values from the user message.
            data_source = (
                "database"
                if any(s.route == "sql" for s in workflow.steps[:index])
                else "inline"
            )

        plan.append(
            PlanItem(
                route=step.route,
                task=step.task,
                data_source=data_source,
            )
        )

    return plan


_TASK_WS_RE = re.compile(r"\s+")
_NUMERIC_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _normalized_task(task: str) -> str:
    return _TASK_WS_RE.sub(" ", task.strip().lower())


def _validate_plan(
    plan: list[PlanItem], state: SupervisorState
) -> tuple[list[PlanItem], Optional[str]]:
    """Deterministic guardrails for a model-produced plan.

    Enforces clarification-only, drops exact duplicate consecutive steps, keeps
    only chart steps that can actually run, caps the plan, and replaces an
    empty result with a clarification instead of a silent end.
    """
    clarification = next((i for i in plan if i.route == "clarification"), None)
    if clarification is not None:
        return [
            PlanItem(
                route="clarification",
                task=clarification.task or "The request was unclear.",
            )
        ], None

    request = " ".join(
        part
        for part in [latest_user_request(state.messages) or "", *clarification_answers(state.messages)]
        if part
    )
    # An inline chart must supply at least two values, otherwise there is
    # nothing to plot and extraction can only fail or invent numbers.
    enough_numbers = len(_NUMERIC_TOKEN_RE.findall(request)) >= 2

    deduped: list[PlanItem] = []
    for item in plan:
        if (
            deduped
            and deduped[-1].route == item.route
            and _normalized_task(deduped[-1].task) == _normalized_task(item.task)
        ):
            continue
        deduped.append(item)

    valid: list[PlanItem] = []
    for index, item in enumerate(deduped):
        if item.route == "visu":
            if item.data_source == "inline" and not enough_numbers:
                continue
            if item.data_source != "inline":
                if not any(p.route == "sql" for p in deduped[:index]):
                    continue
                item.data_source = "database"
        valid.append(item)

    kept, dropped = valid[:MAX_PLAN_STEPS], valid[MAX_PLAN_STEPS:]
    note = None
    if dropped:
        asks = "; ".join(item.task for item in dropped)
        note = (
            f"I focused on the first {MAX_PLAN_STEPS} of {len(valid)} requested "
            f"steps. Not covered: {asks}."
        )

    if not kept:
        task = request or state.current_task or "The request was unclear."
        return [PlanItem(route="clarification", task=task)], note

    return kept, note


def get_workflow_plan(context: dict, model, max_attempts: int = 2) -> list[PlanItem]:
    """One LLM call decomposes the request into an ordered plan.

    Raises LLMOutageError when no provider is usable; returns [] on a schema
    failure so the caller can fall back to a single decision.
    """
    last_class: str | None = None

    for _attempt in range(1, max_attempts + 1):
        classes: list[str | None] = []

        try:
            structured = model.with_structured_output(
                WorkflowPlan, method="function_calling"
            )
            raw = structured.invoke(build_planning_prompt(context))
            workflow = (
                raw if isinstance(raw, WorkflowPlan)
                else WorkflowPlan.model_validate(raw)
            )
            return _plan_from_workflow(workflow)
        except TurnBudgetExceeded:
            raise
        except Exception as e:
            classes.append(classify_llm_error(e))

        last_class = (
            "auth" if "auth" in classes
            else "transient" if "transient" in classes
            else None
        )
        if last_class in ("auth", "transient"):
            break

    if last_class == "auth":
        logger.critical("planner_llm_outage")
        raise LLMOutageError("auth")
    if last_class == "transient":
        logger.error("planner_llm_transient_outage")
        raise LLMOutageError("transient")

    logger.error("workflow_planning_failed")
    return []


def _complete_current_step(
    plan: list[PlanItem], result: SpecialistResult
) -> list[PlanItem]:
    """Write the just-finished result into the first pending step."""
    for item in plan:
        if item.status == "pending":
            item.status = "done" if result.status == "done" else "failed"
            item.result_summary = result.summary
            item.issue = result.issue
            item.structured_data = result.structured_data
            break
    return plan


def _apply_clarification_cap(
    plan: list[PlanItem], state: SupervisorState
) -> list[PlanItem]:
    if state.clarification_count < MAX_CLARIFICATIONS_PER_TURN:
        return plan
    for item in plan:
        if item.route == "clarification":
            item.route = "convo"
            item.task = (
                "The request is still unclear after asking for clarification. "
                "Tell the user plainly that you could not determine what they "
                "need, and ask them to rephrase with more specifics. "
                "Do not invent an answer."
            )
    return plan


def _rows_upstream(plan: list[PlanItem], idx: int) -> list[dict] | None:
    """The nearest done sql step's chartable rows before index ``idx``."""
    for prev in reversed(plan[:idx]):
        if prev.route == "sql" and prev.status == "done" and prev.structured_data:
            return prev.structured_data
    return None


def _maybe_insert_visu(
    plan: list[PlanItem], state: SupervisorState
) -> list[PlanItem]:
    """Additive backstop: add one visu step when the user asked for a chart and
    an upstream sql step already produced rows, in case the planner omitted it.

    Never re-inserts once any visu exists (in any status), and never inserts
    without rows, so it cannot route to an LLM-dependent visu or loop.
    """
    if any(item.route == "visu" for item in plan):
        return plan
    if not _user_wants_visualization(state):
        return plan

    if _rows_upstream(plan, len(plan)) is not None:
        plan.append(
            PlanItem(
                route="visu",
                data_source="database",
                task=(
                    "Create a clear chart from the structured_data of the "
                    "previous result."
                ),
            )
        )
    return plan


def _skip_blocked_steps(plan: list[PlanItem]) -> list[PlanItem]:
    """Skip a database-backed visu that can never get its data.

    A visu still waiting on a pending sql step is left alone; once every
    upstream sql step has finished, a missing row set means no data is coming.
    Inline visus carry their own values and are never skipped here.
    """
    for index, item in enumerate(plan):
        if (
            item.status != "pending"
            or item.route != "visu"
            or item.data_source == "inline"
        ):
            continue

        upstream_sql = [p for p in plan[:index] if p.route == "sql"]
        if any(p.status == "pending" for p in upstream_sql):
            continue
        if not upstream_sql or _rows_upstream(plan, index) is None:
            item.status = "skipped"
            item.issue = "no_data_for_chart"
            item.result_summary = (
                "The chart was skipped because the data was unavailable."
            )
    return plan


def _skip_unbudgeted_visu(
    plan: list[PlanItem], state: SupervisorState
) -> list[PlanItem]:
    """Backstop: once the turn budget is spent, do not route to a visu that
    would need an LLM call. A visu that charts existing structured_data is
    deterministic and remains allowed; an inline visu (or one with no rows)
    is not."""
    budget = get_budget()
    if budget is None or not budget.exhausted():
        return plan

    for index, item in enumerate(plan):
        if item.route != "visu" or item.status != "pending":
            continue

        deterministic = (
            item.data_source != "inline"
            and _rows_upstream(plan, index) is not None
        )
        if not deterministic:
            item.status = "failed"
            item.issue = "budget_exceeded"
            item.result_summary = (
                "This request could not be completed within the allowed "
                "budget for a single turn."
            )
    return plan


def _next_pending(plan: list[PlanItem]) -> PlanItem | None:
    for item in plan:
        if item.status == "pending":
            return item
    return None


def _next_plan_decision(plan: list[PlanItem]) -> SupervisorDecision:
    item = _next_pending(plan)
    if item is None:
        return _end_decision()
    return SupervisorDecision(next=item.route, current_task=item.task)


_CHART_TYPE = r"(?:bar|pie|line|scatter|histogram|donut|doughnut)"


_CHART_EXCLUSIONS = (
    "org chart",
    "chart of accounts",
    "bar association",
    "bar exam",
    "plot of land",
    "plot twist",
    "graph database",
    "graph theory",
    "pie eating",
)
_CHART_EXCLUSION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _CHART_EXCLUSIONS) + r")\b",
    re.IGNORECASE,
)

# Unambiguous chart commands that override an exclusion phrase.
_CHART_HARD_RE = re.compile(
    r"\bvisuali[sz]e\b"
    rf"|\b{_CHART_TYPE}\s+(?:chart|graph|plot)\b"
    rf"|\bas\s+an?\s+{_CHART_TYPE}\b",
    re.IGNORECASE,
)

_CHART_INTENT_RE = re.compile(
    r"\bvisuali[sz]e\b"
    rf"|\b{_CHART_TYPE}\s+(?:chart|graph|plot)\b"
    r"|\b(?:chart|graph|plot|pie|pies|bar|bars)\b"
    rf"|\bas\s+an?\s+(?:{_CHART_TYPE}|chart|graph|plot)\b"
    rf"|\b(?:make|create|draw|generate|show me|give me)\b[^.?!]{{0,30}}"
    rf"\b(?:chart|graph|plot|{_CHART_TYPE})\b",
    re.IGNORECASE,
)


def _user_wants_visualization(state: SupervisorState) -> bool:
    """
    Lightweight chart-intent check.

    Looks at the latest request plus any clarification answers, so an answer
    like "as a pie chart" still counts even though latest_user_request
    deliberately skips tagged clarification answers. Known non-visual
    collocations ("org chart", "plot of land", ...) are excluded unless an
    unambiguous chart command is also present.
    """
    if not state.messages:
        return False

    parts = [latest_user_request(state.messages), *clarification_answers(state.messages)]
    text = " ".join(part for part in parts if part).lower()
    if not text:
        return False

    if _CHART_EXCLUSION_RE.search(text):
        return bool(_CHART_HARD_RE.search(text))

    return bool(_CHART_INTENT_RE.search(text))


def gather_context(
    state: SupervisorState,
    task_history: list[TaskRecord],
    user_id: str | None = None,
) -> dict:
    known_facts = ""
    if user_id:
        try:
            known_facts = format_facts_for_prompt(user_id)
        # Long-term memory is an enhancement; a DB failure must not block
        # routing, so this deliberately degrades instead of narrowing.
        except Exception as e:
            logger.warning(
                "known_facts_lookup_failed_continuing_without_it",
                extra={"error": str(e)},
            )

    logger.debug(
        "gather_context",
        extra={
            "n_messages": len(state.messages),
            "last_result": repr(state.last_result),
            "n_task_history": len(task_history),
            "has_known_facts": bool(known_facts),
        },
    )

    return {
        "messages": state.messages,
        "last_result": state.last_result,
        "task_history": task_history,
        "known_facts": known_facts,
        "conversation_summary": state.conversation_summary,
    }


def build_prompt(context: dict, previous_error: str | None = None) -> list:
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]

    if context.get("conversation_summary"):
        messages.append(
            SystemMessage(
                content=f"[Summary of earlier conversation]: "
                        f"{context['conversation_summary']}"
            )
        )
    
    if context.get("known_facts"):
        messages.append(SystemMessage(content=context["known_facts"]))

    if context["last_result"] is not None:
        lr = context["last_result"]
        messages.append(
            SystemMessage(
                content=(
                    f"Last specialist result — source: {lr.source}, "
                    f"status: {lr.status}, issue: {lr.issue or 'none'}\n"
                    f"Summary: {lr.summary}"
                )
            )
        )

    recent = context["messages"][-6:]
    messages.extend(recent)

    if previous_error:
        messages.append(
            SystemMessage(
                content=(
                    f"Your previous response failed validation: {previous_error}. "
                    "Correct it and respond again in the required schema."
                )
            )
        )

    return messages


def get_supervisor_decision(context: dict, model, max_attempts: int = 2) -> Optional[SupervisorDecision]:
    last_error: str | None = None
    last_class: str | None = None  # most recent attempt only

    for attempt in range(1, max_attempts + 1):
        prompt = build_prompt(context, previous_error=last_error)
        classes: list[str | None] = []
        errors: list[str] = []

        # json_mode is unusable with Groq (it requires the literal word "json"
        # in the prompt), so function_calling is the only structured method.
        try:
            structured = model.with_structured_output(
                SupervisorDecision, method="function_calling"
            )
            raw = structured.invoke(prompt)
            return SupervisorDecision.model_validate(raw)
        except TurnBudgetExceeded:
            # Not a schema/provider failure: don't retry, don't classify.
            raise
        except Exception as e:
            errors.append(f"function_calling: {e}")
            classes.append(classify_llm_error(e))

        last_error = " | ".join(errors)
        last_class = (
            "auth" if "auth" in classes
            else "transient" if "transient" in classes
            else None
        )

        logger.debug(
            "structured_output_attempt_failed",
            extra={"attempt": attempt, "class": last_class or "schema", "error": last_error[:200]},
        )

        if last_class in ("auth", "transient"):
            break  # don't retry auth; don't immediately re-hit a 429

    if last_class == "auth":
        logger.critical("supervisor_llm_outage")
        raise LLMOutageError("auth")
    if last_class == "transient":
        logger.error("supervisor_llm_transient_outage")
        raise LLMOutageError("transient")

    logger.error("all_structured_output_attempts_failed")
    return None


def get_current_turn(state: SupervisorState) -> int:
    return state.turn_count


def map_to_state(decision: SupervisorDecision) -> dict:
    """Clear last_result unless routing to visu/end, which still need structured_data."""
    if decision.next in ("end", "visu"):
        return {
            "next": decision.next,
            "current_task": decision.current_task,
        }

    return {
        "next": decision.next,
        "current_task": decision.current_task,
        "last_result": None,
    }