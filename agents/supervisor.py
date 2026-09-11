from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from state.state import SupervisorState, TaskRecord, SpecialistResult
from state.structure_output import SupervisorDecision
from services.llm import llm
from services.memory import format_facts_for_prompt
from services.errors import classify_llm_error

logger = logging.getLogger("supervisor")

MAX_HOPS_PER_TURN = 6
MAX_SAME_ROUTE_PER_TURN = 2

SUPERVISOR_SYSTEM_PROMPT = """You are the routing supervisor for a company intelligence assistant.

Decide which specialist should handle the latest user request.

Routes:
- rag: internal company documents, policies, procedures. DEFAULT for organizational questions.
- research: external/public information. Only when the user explicitly asks to research, look up, or find external information. Never for organizational topics.
- sql: questions that need live structured data (counts, sums, specific records) — this includes chart/visualization requests whose underlying data hasn't been fetched yet.
- visu: turns structured_data that already exists (from a previous specialist result) into a chart.
- convo: you can answer directly (definitions, small talk, clarifying your own prior answer, or synthesizing when a specialist result needs light rephrasing).
- clarification: you genuinely don't know WHAT the user is asking for — the request itself is unclear, not just which specialist should handle it.
- end: nothing further needs to happen.

Chart requests are naturally two steps: fetch the data (sql), then draw it (visu) once structured_data exists. If a chart is requested and structured_data doesn't exist yet, route to sql. That sequencing is normal and expected — it is not ambiguity.

Examples:
- "Chart the total sales by region." (no structured_data yet) → sql
- "Show a pie chart of regional sales share." (no structured_data yet) → sql
- "Now chart that as a bar graph." (structured_data exists from prior turn) → visu
- "What's our remote work policy?" → rag
- "Look up the latest EU AI Act news." → research

If the user say goodbye or anything similar respond to it in convo instead of directly routing to end
For convo, put a short pre-summary of relevant context in current_task.
For all other routes, current_task must be a concise actionable task description.
For end, current_task can be empty.

"""

_OUTAGE_TASK = (
    "Explain that the assistant is temporarily unavailable due to a service "
    "issue, not because the request was unclear. Ask them to try again later. "
    "Do not invent an answer."
)

def supervisor_agent(state: SupervisorState, config: RunnableConfig) -> dict:
    """
    1. Record the previous specialist result into task_history (if any).
    2. Try a deterministic decision.
    3. Fall back to LLM only when necessary.
    4. Always apply post-guards.
    5. Map the final decision into a state update.
    """
    task_history = list(state.task_history)

    # ----- 1. Record previous work -----
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

    # ----- 2. Deterministic first -----
    decision = deterministic_decision(state, task_history)

    if decision is None:
        user_id = (config.get("configurable") or {}).get("user_id")
        try:
            context = gather_context(state, task_history, user_id)
            decision = get_supervisor_decision(context, llm())
        except Exception as e:
            raise RuntimeError(f"Supervisor LLM call failed: {e}") from e

    # ----- 4. Post-guards (always) -----
    decision = post_decision_guards(state, decision, task_history)
    decision = enforce_task_history_guard(state, decision, task_history)

    logger.debug(
        "final_decision",
        extra={"next": decision.next, "current_task": decision.current_task},
    )

    # ----- 5. Map to state update -----
    update = map_to_state(decision)
    update["task_history"] = task_history
    return update


def deterministic_decision(
    state: SupervisorState,
    task_history: list[TaskRecord],
) -> Optional[SupervisorDecision]:
    current_turn = get_current_turn(state)
    turn_history = [r for r in task_history if r.turn == current_turn]
    lr: Optional[SpecialistResult] = state.last_result

    # Hard hop limit
    if len(turn_history) >= MAX_HOPS_PER_TURN:
        logger.warning(
            "hop_limit_reached",
            extra={"limit": MAX_HOPS_PER_TURN, "turn": current_turn},
        )
        return SupervisorDecision(next="end", current_task="")
   
    if lr is not None:

        '''
        if (
            lr.status == "done"
            and lr.structured_data
            and _user_wants_visualization(state)
        ):
            logger.debug("structured_data_plus_visualization_intent")
            return SupervisorDecision(
                next="visu",
                current_task="Create a clear chart from the structured_data of the previous result."
            )

        # Normal success → force end (prevents loops)
        if lr.status == "done":
            logger.debug("last_result_done_forcing_end")
            return SupervisorDecision(next="end", current_task="")
        '''

        if lr.source == "rag" and lr.issue == "rag_unavailable":
            return SupervisorDecision(
                next="convo",
                current_task="Tell the user internal search is temporarily unavailable. Do not invent an answer.",
            )
        
        # Terminal permission failure
        if lr.issue == "permission_denied":
            already_explained = any(
                r.route == "convo" and r.status == "done" for r in turn_history
            )
            if already_explained:
                logger.debug("permission_denied_already_explained")
                return SupervisorDecision(next="end", current_task="")

            logger.debug("permission_denied_routing_to_convo")
            return SupervisorDecision(
                next="convo",
                current_task=(
                    "Explain the permission restriction clearly and stop. "
                    "Do not offer alternatives that require the same data."
                ),
            )

        # RAG found nothing
        if lr.source == "rag" and lr.issue == "no_matching_documents":
            logger.debug("rag_no_matching_documents")
            return SupervisorDecision(
                next="convo",
                current_task=(
                    "Tell the user no internal documents matched this request and stop. "
                    "Do not invent external research."
                ),
            )

        # Same specialist already failed/partial once → explain & stop
        same_route_failures = [
            r for r in turn_history
            if r.route == lr.source and r.status in ("partial", "failed")
        ]
        if len(same_route_failures) >= 1:
            logger.debug("route_already_failed_or_partial", extra={"route": lr.source})
            return SupervisorDecision(
                next="convo",
                current_task=(
                    f"Explain that the {lr.source} specialist could not fully answer "
                    f"and what the limitation was. Then stop."
                ),
            )

    if not state.messages:
        return SupervisorDecision(next="end", current_task="")

    return None  # residual → LLM


def _user_wants_visualization(state: SupervisorState) -> bool:
    """Very lightweight intent check — only looks at the latest human message."""
    if not state.messages:
        return False

    last_human_content = None
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage):
            last_human_content = m.content
            break

    if last_human_content is None:
        return False

    # content can be str or list (multimodal). Normalize to a single lowercase string.
    if isinstance(last_human_content, list):
        text_parts = []
        for part in last_human_content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        text = " ".join(text_parts).lower()
    else:
        text = str(last_human_content).lower()

    keywords = ["chart", "graph", "plot", "visualize", "visualise", "bar", "pie", "line chart"]
    return any(k in text for k in keywords)


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
            return SupervisorDecision(next="end", current_task="")

    # After a successful result, never send back to the same specialist
    if (
        state.last_result
        and state.last_result.status == "done"
        and decision.next == state.last_result.source
    ):
        logger.warning(
            "refused_reroute_to_finished_specialist",
            extra={"route": decision.next},
        )
        return SupervisorDecision(next="end", current_task="")

    # Only one clarification attempt per turn
    if decision.next == "clarification":
        clar_count = sum(1 for r in turn_history if r.route == "clarification")
        # Note: clarification itself is not recorded in task_history in the
        # current codebase; if you later record it, this becomes active.
        # For now we keep the check as a future-proof guard.
        if clar_count >= 1:
            logger.warning("clarification_already_used")
            return SupervisorDecision(next="end", current_task="")

    return decision


def _normalize_task(task: str) -> str:
    return " ".join(task.lower().split())


def enforce_task_history_guard(
    state: SupervisorState,
    decision: SupervisorDecision,
    task_history: list[TaskRecord],
) -> SupervisorDecision:
    """
    Block the exact same completed task on the same route within the same turn.
    (Preserved from the original implementation, with normalized comparison.)
    """
    if decision.next == "end":
        return decision

    if decision.next not in ("rag", "convo", "sql", "research", "visu"):
        return decision

    current_turn = get_current_turn(state)
    proposed_task = (decision.current_task or "").strip()
    if not proposed_task:
        return decision

    normalized_proposed = _normalize_task(proposed_task)

    for record in reversed(task_history):
        if record.turn != current_turn:
            continue
        if record.status != "done":
            continue
        if record.route != decision.next:
            continue
        if _normalize_task(record.task) != normalized_proposed:
            continue

        logger.warning(
            "blocked_duplicate_completed_task",
            extra={"route": decision.next, "task": proposed_task},
        )
        return SupervisorDecision(next="end", current_task="")

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
        except Exception as e:
            # Long-term memory is a nice-to-have, not a hard dependency, same
            # reasoning as Finalize's write-side handling. A DB hiccup here
            # should degrade this turn's context, not crash the entire graph
            # on what might just be "hello".
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
    }


def build_prompt(context: dict, previous_error: str | None = None) -> list:
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]

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

    recent = context["messages"][-6:] if len(context["messages"]) > 6 else context["messages"]
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


def get_supervisor_decision(context: dict, model, max_attempts: int = 2) -> SupervisorDecision:
    last_error: str | None = None
    last_class: str | None = None  # most recent attempt only

    for attempt in range(1, max_attempts + 1):
        prompt = build_prompt(
            context,
            previous_error=None if last_class else last_error,
        )
        classes: list[str | None] = []
        errors: list[str] = []

        for method in ("json_mode", "function_calling"):
            try:
                structured = model.with_structured_output(SupervisorDecision, method=method)
                raw = structured.invoke(prompt)
                return SupervisorDecision.model_validate(raw)
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
        return SupervisorDecision(next="convo", current_task=_OUTAGE_TASK)
    if last_class == "transient":
        logger.error("supervisor_llm_transient_outage")
        return SupervisorDecision(next="convo", current_task=_OUTAGE_TASK)

    logger.error("all_structured_output_attempts_failed_fallback_to_clarification")
    return SupervisorDecision(
        next="clarification",
        current_task="Could not determine the best specialist. Please rephrase the request.",
)


def get_current_turn(state: SupervisorState) -> int:
    return state.turn_count


def map_to_state(decision: SupervisorDecision) -> dict:
    """
    Clear last_result when routing to a specialist that should start fresh.
    Keep it for visu/end so downstream nodes can still read structured_data.
    """
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