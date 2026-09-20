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

MAX_HOPS_PER_TURN = 6
MAX_SAME_ROUTE_PER_TURN = 2
MAX_CLARIFICATIONS_PER_TURN = 1
MAX_MULTI_INTENT_HOPS = 1
MAX_PLAN_STEPS = 3


def _end_decision() -> SupervisorDecision:
    return SupervisorDecision(next="end", current_task="")

SUPERVISOR_SYSTEM_PROMPT = """You are the routing supervisor for a company intelligence assistant.

Decide the single best next route for the latest user request.

### Routes
- rag: Internal company documents, policies, procedures, lessons-learned, and organizational knowledge. Choose this when the answer lives in a document: what a policy says, what a procedure is, or what was learned from a past project. Do NOT use it for concrete records in the operational database.
- research: External/public information, including current or time-sensitive facts (today's weather, the latest price, who currently holds a position, recent news). Use it whenever the answer is not company-internal and may have changed since training, even if the user did not say "research". Never use it for company-internal topics.
- sql: Live structured data from the operational database — counts, sums, lists, filters, rankings, and lookups of specific records such as one person's department, salary, or employee ID, a sale's amount/date, or a credential row. Choose sql whenever the answer is a field value in a table, even when the question names a person and is phrased like "What is <name>'s <field>?" or "Find the <id> for <name>".
- visu: Only when structured_data already exists from a previous result and the user wants a chart/graph/plot.
- convo: You can answer directly (greetings, goodbyes, definitions, small talk, clarifying your own previous answer, or light synthesis).
- clarification: The request itself is genuinely unclear — you do not understand what the user wants. Do NOT use this just because the request might need multiple steps.
- end: Nothing further needs to happen.

### Decision rules
- Chart requests without existing structured_data → sql (this is normal, not ambiguity).
- Prefer rag over research for anything internal.
- Prefer convo over clarification whenever the request is understandable.
- Goodbyes, thanks, and small talk → convo.
- Facts about a specific record/entity in the database (a named person's department, salary, or ID; a sale row; a count or list of rows) → sql, even when phrased conversationally.
- Questions about what a document/policy/procedure says, or what was learned from past projects → rag.
- Questions about current, latest, or time-sensitive external facts (current CEO/president, latest price, today's weather, recent news) → research, even without an explicit "research" verb.
- Definitions and general knowledge (e.g. "What does SQL stand for?", "What does RAG mean?") → convo, never sql.
- If the latest user message contains multiple distinct asks and the last specialist result only answered part of it, route to the appropriate specialist for the remaining part instead of ending.
- Focus almost exclusively on the latest user message and the latest specialist result. Ignore older conversation history unless it is directly needed to understand the current request.

### sql vs rag
Decide by where the answer comes from:
- A field inside a database table (employee, department, salary, sales, credentials, row counts) → sql.
- A document, policy, procedure, or lessons-learned note → rag.

### Examples
- "How many employees are there?" → sql
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
- visu: build a chart from data a previous sql step produced. Include it as a step after sql when the user asks for a chart/graph/plot.
- convo: answer directly (greetings, definitions, small talk, light synthesis).
- clarification: the request is genuinely unclear. Use this as the ONLY step.

### Rules
- One step per distinct ask, in the order they should run.
- A single-intent request has exactly one step.
- A two-part request ("how many employees, and what does the policy say?") has two steps: sql then rag.
- A chart over database data is two steps: sql then visu.
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

    user_id = (config.get("configurable") or {}).get("user_id")

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
                plan = [PlanItem(route=fallback.next, task=fallback.current_task)]
            elif fallback is None and not outage:
                request = latest_user_request(state.messages) or ""
                if mentions_sensitive_data(request):
                    plan = [PlanItem(route="sql", task=request)]
                else:
                    plan = [PlanItem(route="convo", task=_OUTAGE_TASK)]

        plan = _apply_clarification_cap(plan, state)

    plan = _maybe_insert_visu(plan, state)

    decision = _end_decision() if outage else _next_plan_decision(plan)

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


_QUESTION_WORD_RE = re.compile(
    r"\b(how many|how much|what|which|who|where|when|why)\b",
    re.IGNORECASE,
)
_MULTI_INTENT_SEPARATOR_RE = re.compile(r"(,|;|\band\b|\bthen\b|\balso\b)", re.IGNORECASE)


def _has_multi_intent(state: SupervisorState) -> bool:
    """Cheap check for two distinct asks in the latest user request.

    Requires at least two question clauses so single-intent conjunctions
    ("employees and departments") do not trigger an extra hop.
    """
    text = (latest_user_request(state.messages) or "").strip()
    if not text:
        return False
    question_words = _QUESTION_WORD_RE.findall(text)
    if len(question_words) < 2:
        return False
    return text.count("?") >= 2 or bool(_MULTI_INTENT_SEPARATOR_RE.search(text))


def _should_allow_multi_intent_hop(state: SupervisorState) -> bool:
    lr = state.last_result
    return (
        lr is not None
        and lr.status == "done"
        and state.multi_intent_hops < MAX_MULTI_INTENT_HOPS
        and _has_multi_intent(state)
    )


def _failure_explanation(result: SpecialistResult) -> str:
    """A user-safe limitation string for the convo agent.

    Prefers the specialist's synthesized summary and never includes the raw
    issue detail (which can contain database/provider errors).
    """
    summary = (result.summary or "").strip()
    if summary:
        return summary[:300]
    code = (result.issue or "").split(":", 1)[0].strip()
    return code or "the specialist could not complete the request"


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
    steps = list(workflow.steps)[:MAX_PLAN_STEPS]

    clarification = next((s for s in steps if s.route == "clarification"), None)
    if clarification is not None:
        return [
            PlanItem(
                route="clarification",
                task=clarification.task or "The request was unclear.",
            )
        ]

    return [
        PlanItem(route=step.route, task=step.task)
        for step in steps
        if step.task and step.task.strip()
    ]


def get_workflow_plan(context: dict, model, max_attempts: int = 2) -> list[PlanItem]:
    """One LLM call decomposes the request into an ordered plan.

    Raises LLMOutageError when no provider is usable; returns [] on a schema
    failure so the caller can fall back to a single decision.
    """
    last_class: str | None = None

    for _attempt in range(1, max_attempts + 1):
        classes: list[str | None] = []

        for method in ("function_calling", "json_mode"):
            try:
                structured = model.with_structured_output(WorkflowPlan, method=method)
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


def _maybe_insert_visu(
    plan: list[PlanItem], state: SupervisorState
) -> list[PlanItem]:
    if any(item.route == "visu" and item.status == "pending" for item in plan):
        return plan
    if not _user_wants_visualization(state):
        return plan
    has_rows = any(
        item.route == "sql" and item.status == "done" and item.structured_data
        for item in plan
    )
    if has_rows:
        plan.append(
            PlanItem(
                route="visu",
                task=(
                    "Create a clear chart from the structured_data of the "
                    "previous result."
                ),
            )
        )
    return plan


def _next_plan_decision(plan: list[PlanItem]) -> SupervisorDecision:
    for item in plan:
        if item.status == "pending":
            return SupervisorDecision(next=item.route, current_task=item.task)
    return _end_decision()


def deterministic_decision(
    state: SupervisorState,
    task_history: list[TaskRecord],
) -> Optional[SupervisorDecision]:
    current_turn = get_current_turn(state)
    turn_history = [r for r in task_history if r.turn == current_turn]
    lr: Optional[SpecialistResult] = state.last_result

    if len(turn_history) >= MAX_HOPS_PER_TURN:
        logger.warning(
            "hop_limit_reached",
            extra={"limit": MAX_HOPS_PER_TURN, "turn": current_turn},
        )
        return _end_decision()

    if (
        lr is not None
        and lr.status == "done"
        and lr.structured_data
        and _user_wants_visualization(state)
    ):
        logger.debug("structured_data_plus_visualization_intent")
        return SupervisorDecision(
            next="visu",
            current_task="Create a clear chart from the structured_data of the previous result."
        )

    budget = get_budget()
    if budget is not None and budget.exhausted():
        logger.warning("turn_budget_exhausted_forcing_end")
        return _end_decision()

    if lr is not None:
        # Normal success → force end, unless this is a multi-part request with
        # an unused bounded hop (then let the LLM route the remaining part).
        if lr.status == "done":
            if _should_allow_multi_intent_hop(state):
                logger.debug("multi_intent_allowing_extra_hop")
                return None
            logger.debug("last_result_done_forcing_end")
            return _end_decision()
        

        if lr.source == "rag" and lr.issue == "rag_unavailable":
            return SupervisorDecision(
                next="convo",
                current_task="Tell the user internal search is temporarily unavailable. Do not invent an answer.",
            )
        
        if lr.issue == "permission_denied":
            already_explained = any(
                r.route == "convo" and r.status == "done" for r in turn_history
            )
            if already_explained:
                logger.debug("permission_denied_already_explained")
                return _end_decision()

            logger.debug("permission_denied_routing_to_convo")
            return SupervisorDecision(
                next="convo",
                current_task=(
                    "Explain the permission restriction clearly and stop. "
                    "Do not offer alternatives that require the same data."
                ),
            )

        if lr.source == "rag" and lr.issue == "no_matching_documents":
            logger.debug("rag_no_matching_documents")
            return SupervisorDecision(
                next="convo",
                current_task=(
                    "Tell the user no internal documents matched this request and stop. "
                    "Do not invent external research."
                ),
            )

        same_route_failures = [
            r for r in turn_history
            if r.route == lr.source and r.status in ("partial", "failed")
        ]
        # The failure just recorded is always the last matching record, so drop
        # it: the first failure returns residual to the LLM (one retry) and
        # only a repeat failure routes to convo.
        prior_failures = same_route_failures[:-1]
        if prior_failures:
            reason = _failure_explanation(lr)
            logger.debug("route_already_failed_or_partial", extra={"route": lr.source})
            return SupervisorDecision(
                next="convo",
                current_task=(
                    f"Explain that the {lr.source} specialist could not fully "
                    f"answer. The reason reported was: {reason}. "
                    f"Do not invent details. Then stop."
                ),
            )

    if not state.messages:
        return _end_decision()

    return None  # residual → LLM


_CHART_TYPE = r"(?:bar|pie|line|scatter|histogram|donut|doughnut)"

# Phrases that mention chart vocabulary but are not visualization requests.
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

# Broad chart vocabulary. "line" is intentionally absent (too ambiguous); it
# only counts via "line chart"/"line graph"/"as a line".
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


def post_decision_guards(
    state: SupervisorState,
    decision: SupervisorDecision,
    task_history: list[TaskRecord],
) -> SupervisorDecision:
    current_turn = get_current_turn(state)
    turn_history = [r for r in task_history if r.turn == current_turn]

    # Cap same route per turn
    if decision.next in ("rag", "sql", "research", "visu", "convo"):
        same_route_count = sum(1 for r in turn_history if r.route == decision.next)
        if same_route_count >= MAX_SAME_ROUTE_PER_TURN:
            logger.warning(
                "same_route_cap_hit",
                extra={"route": decision.next, "cap": MAX_SAME_ROUTE_PER_TURN},
            )
            return _end_decision()

    if (
        state.last_result
        and state.last_result.status == "done"
        and decision.next == state.last_result.source
    ):
        logger.warning(
            "refused_reroute_to_finished_specialist",
            extra={"route": decision.next},
        )
        return _end_decision()

    # Only one clarification attempt per turn
    if decision.next == "clarification":
        if state.clarification_count >= MAX_CLARIFICATIONS_PER_TURN:
            logger.warning(
                "clarification_cap_hit",
                extra={"count": state.clarification_count,
                       "cap": MAX_CLARIFICATIONS_PER_TURN},
            )
            # Route to convo, not end, so the user still gets a reply.
            return SupervisorDecision(
                next="convo",
                current_task=(
                    "The request is still unclear after asking for clarification. "
                    "Tell the user plainly that you could not determine what they "
                    "need, and ask them to rephrase with more specifics. "
                    "Do not invent an answer."
                ),
            )

    return decision


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
        prompt = build_prompt(
            context,
            previous_error=None if last_class else last_error,
        )
        classes: list[str | None] = []
        errors: list[str] = []

        for method in ("function_calling","json_mode"):
            try:
                structured = model.with_structured_output(SupervisorDecision, method=method)
                raw = structured.invoke(prompt)
                return SupervisorDecision.model_validate(raw)
            except TurnBudgetExceeded:
                # Not a schema/provider failure: don't retry, don't classify.
                raise
            except Exception as e:
                errors.append(f"{method}: {e}")
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