from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from state.state import SupervisorState, TaskRecord, SpecialistResult
from state.structure_output import SupervisorDecision, ClarificationOutput
from services.message_utils import clarification_answers, latest_user_request
from services.llm import llm
from services.memory import format_facts_for_prompt
from services.errors import classify_llm_error

logger = logging.getLogger("supervisor")

MAX_HOPS_PER_TURN = 6
MAX_SAME_ROUTE_PER_TURN = 2
MAX_CLARIFICATIONS_PER_TURN = 1

SUPERVISOR_SYSTEM_PROMPT = """You are the routing supervisor for a company intelligence assistant.

Decide the single best next route for the latest user request.

### Routes
- rag: Internal company documents, policies, procedures, or organizational knowledge. Prefer this for almost all internal questions.
- research: External/public information. ONLY when the user explicitly asks to research, look up, or find external information. Never use for company topics.
- sql: Needs live structured data (counts, sums, lists, filters, rankings, or any chart/visualization whose data has not been fetched yet).
- visu: Only when structured_data already exists from a previous result and the user wants a chart/graph/plot.
- convo: You can answer directly (greetings, goodbyes, definitions, small talk, clarifying your own previous answer, or light synthesis).
- clarification: The request itself is genuinely unclear — you do not understand what the user wants. Do NOT use this just because the request might need multiple steps.
- end: Nothing further needs to happen.

### Decision rules
- Chart requests without existing structured_data → sql (this is normal, not ambiguity).
- Prefer rag over research for anything internal.
- Prefer convo over clarification whenever the request is understandable.
- Goodbyes, thanks, and small talk → convo.
- When in doubt between rag and sql, prefer the one that best matches the user's actual need.
-Focus almost exclusively on the latest user message and the latest specialist result, Ignore older conversation history unless it is directly needed to understand the current request

### current_task
- convo → short pre-summary of relevant context
- all other routes → concise, actionable task description
- end → leave empty


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

    decision = deterministic_decision(state, task_history)

    if decision is None:
        user_id = (config.get("configurable") or {}).get("user_id")
        try:
            context = gather_context(state, task_history, user_id)
            decision = get_supervisor_decision(context, llm())
        except Exception as e:
            raise RuntimeError(f"Supervisor LLM call failed: {e}") from e

    decision = post_decision_guards(state, decision, task_history)
    decision = enforce_task_history_guard(state, decision, task_history)

    logger.debug(
        "final_decision",
        extra={"next": decision.next, "current_task": decision.current_task},
    )

    update = map_to_state(decision)
    update["task_history"] = task_history

    if decision.next == "clarification":
        update["clarification_question"] = _generate_clarification_question(decision)

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
    except Exception as e:
        logger.warning(
            "clarification_question_generation_failed",
            extra={"error": f"{type(e).__name__}: {e}"},
        )
    return "Could you clarify what you'd like me to do?"


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
        return SupervisorDecision(next="end", current_task="")

    if lr is not None:
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
                return SupervisorDecision(next="end", current_task="")

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
    """
    Lightweight chart-intent check.

    Looks at the latest request plus any clarification answers, so an answer
    like "as a pie chart" still counts even though latest_user_request
    deliberately skips tagged clarification answers.
    """
    if not state.messages:
        return False

    parts = [latest_user_request(state.messages), *clarification_answers(state.messages)]
    text = " ".join(part for part in parts if part).lower()
    if not text:
        return False

    if "line chart" in text or "line graph" in text:
        return True
    return bool(
        re.search(r"\b(chart|graph|plot|visuali[sz]e|bar|bars|pie|pies)\b", text)
    )


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


def _normalize_task(task: str) -> str:
    return " ".join(task.lower().split())


def enforce_task_history_guard(
    state: SupervisorState,
    decision: SupervisorDecision,
    task_history: list[TaskRecord],
) -> SupervisorDecision:
    """Block the exact same completed task on the same route within the same turn."""
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

        for method in ("function_calling","json_mode"):
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