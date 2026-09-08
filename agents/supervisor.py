from __future__ import annotations
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError
from state.state import SupervisorState, TaskRecord, SpecialistResult
from state.structure_output import SupervisorDecision
from services.llm import llm

MAX_HOPS_PER_TURN = 6
MAX_SAME_ROUTE_PER_TURN = 2

SUPERVISOR_SYSTEM_PROMPT = """You are the routing supervisor for a company intelligence assistant.
Decide which specialist should handle the latest user request.

Routes:
- rag: internal company documents, policies, procedures. DEFAULT for organizational questions.
- research: external/public information. Only when the user explicitly asks to research, look up, or find external information. Never for organizational topics.
- sql: questions that need live structured data (counts, sums, specific records).
- visu: requests to visualize or chart data. When structured_data already exists from a previous specialist, just describe what to chart.
- convo: you can answer directly (definitions, small talk, clarifying your own prior answer, or synthesizing when a specialist result needs light rephrasing).
- clarification: the request genuinely maps to more than one route and cannot be disambiguated from the message alone.
- end: nothing further needs to happen.

For convo, put a short pre-summary of relevant context in current_task.
For all other routes, current_task must be a concise actionable task description.
For end, current_task can be empty.
"""


def supervisor_agent(state: SupervisorState) -> dict:
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
        print(
            f"[SUPERVISOR] recorded task: turn={current_turn} "
            f"route={task_record.route!r} task={task_record.task!r} "
            f"status={task_record.status!r}"
        )

    # ----- 2. Deterministic first -----
    decision = deterministic_decision(state, task_history)

    if decision is None:
        # ----- 3. LLM residual case -----
        context = gather_context(state, task_history)
        try:
            decision = get_supervisor_decision(context, llm())
        except Exception as e:
            raise RuntimeError(f"Supervisor LLM call failed: {e}") from e

    # ----- 4. Post-guards (always) -----
    decision = post_decision_guards(state, decision, task_history)
    decision = enforce_task_history_guard(state, decision, task_history)

    print(f"[SUPERVISOR] final decision.next={decision.next!r} current_task={decision.current_task!r}")

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
        print(f"[SUPERVISOR] hop limit ({MAX_HOPS_PER_TURN}) reached → end")
        return SupervisorDecision(next="end", current_task="")

    if lr is not None:
        # ---------------------------------------------------------------
        # Special case: successful specialist that produced chartable data
        # and the user is asking for a visualization → allow visu
        # ---------------------------------------------------------------
        if (
            lr.status == "done"
            and lr.structured_data
            and _user_wants_visualization(state)
        ):
            print("[SUPERVISOR] structured_data present + visualization intent → visu")
            return SupervisorDecision(
                next="visu",
                current_task="Create a clear chart from the structured_data of the previous result."
            )

        # Normal success → force end (prevents loops)
        if lr.status == "done":
            print("[SUPERVISOR] last_result status=done → end")
            return SupervisorDecision(next="end", current_task="")

        # Terminal permission failure
        if lr.issue == "permission_denied":
            already_explained = any(
                r.route == "convo" and r.status == "done" for r in turn_history
            )
            if already_explained:
                print("[SUPERVISOR] permission_denied already explained → end")
                return SupervisorDecision(next="end", current_task="")
            print("[SUPERVISOR] permission_denied → convo (explain once)")
            return SupervisorDecision(
                next="convo",
                current_task=(
                    "Explain the permission restriction clearly and stop. "
                    "Do not offer alternatives that require the same data."
                ),
            )

        # RAG found nothing
        if lr.source == "rag" and lr.issue == "no_matching_documents":
            print("[SUPERVISOR] RAG no_matching_documents → convo")
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
            print(f"[SUPERVISOR] {lr.source} already failed/partial → convo")
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
    """Very lightweight intent check  only looks at the latest human message."""
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
        # join any text parts
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
            print(
                f"[SUPERVISOR GUARD] same-route cap ({MAX_SAME_ROUTE_PER_TURN}) "
                f"for {decision.next!r} → end"
            )
            return SupervisorDecision(next="end", current_task="")

    # After a successful result, never send back to the same specialist
    if (
        state.last_result
        and state.last_result.status == "done"
        and decision.next == state.last_result.source
    ):
        print("[SUPERVISOR GUARD] refused re-route to just-finished specialist → end")
        return SupervisorDecision(next="end", current_task="")

    # Only one clarification attempt per turn
    if decision.next == "clarification":
        clar_count = sum(1 for r in turn_history if r.route == "clarification")
        # Note: clarification itself is not recorded in task_history in the
        # current codebase; if you later record it, this becomes active.
        # For now we keep the check as a future-proof guard.
        if clar_count >= 1:
            print("[SUPERVISOR GUARD] clarification already used → end")
            return SupervisorDecision(next="end", current_task="")

    return decision


def enforce_task_history_guard(
    state: SupervisorState,
    decision: SupervisorDecision,
    task_history: list[TaskRecord],
) -> SupervisorDecision:
    """
    Block the exact same completed task on the same route within the same turn.
    (Preserved from the original implementation.)
    """
    if decision.next == "end":
        return decision

    if decision.next not in ("rag", "convo", "sql", "research", "visu"):
        return decision

    current_turn = get_current_turn(state)
    proposed_task = (decision.current_task or "").strip()
    if not proposed_task:
        return decision

    for record in reversed(task_history):
        if record.turn != current_turn:
            continue
        if record.status != "done":
            continue
        if record.route != decision.next:
            continue
        if record.task.strip() != proposed_task:
            continue

        print(
            f"[SUPERVISOR GUARD] blocked duplicate completed task: "
            f"route={decision.next!r}, task={proposed_task!r}"
        )
        return SupervisorDecision(next="end", current_task="")

    return decision


def gather_context(
    state: SupervisorState,
    task_history: list[TaskRecord],
) -> dict:
    print(
        f"[GATHER_CONTEXT] {len(state.messages)} messages, "
        f"last_result={state.last_result!r}, "
        f"task_history={len(task_history)} records"
    )
    return {
        "messages": state.messages,
        "last_result": state.last_result,
        "task_history": task_history,
    }


def build_prompt(context: dict, previous_error: str | None = None) -> list:
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]

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


def get_supervisor_decision(
    context: dict,
    model,
    max_attempts: int = 2,
) -> SupervisorDecision:
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        prompt = build_prompt(context, previous_error=last_error)

        # 1. Prefer json_mode – much more reliable on Groq
        try:
            structured = model.with_structured_output(
                SupervisorDecision, method="json_mode"
            )
            raw = structured.invoke(prompt)
            return SupervisorDecision.model_validate(raw)
        except Exception as e1:
            # 2. Fall back to function_calling
            try:
                structured = model.with_structured_output(
                    SupervisorDecision, method="function_calling"
                )
                raw = structured.invoke(prompt)
                return SupervisorDecision.model_validate(raw)
            except Exception as e2:
                # Capture the most useful error message
                err_msg = str(e2)
                if "tool_use_failed" in err_msg or "Tool choice is required" in err_msg:
                    last_error = (
                        "Model failed to call the required tool. "
                        "You MUST respond with a valid SupervisorDecision object."
                    )
                else:
                    last_error = err_msg
                print(f"[SUPERVISOR] structured-output attempt {attempt} failed: {last_error[:200]}")
                continue

    # Exhausted retries – never crash the graph
    print("[SUPERVISOR] all structured-output attempts failed → fallback to clarification")
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