from state.state import SupervisorState , TaskRecord
from state.structure_output import SupervisorDecision
from pydantic import ValidationError
from langchain_core.messages import SystemMessage, HumanMessage
from services.llm import llm

SUPERVISOR_SYSTEM_PROMPT = """You are the routing supervisor for a company intelligence assistant.
    Based on the conversation, decide which specialist should handle the latest request.
 
    Routes:
    - rag: internal company documents, policies, procedures. DEFAULT for organizational questions.
    - research: external/public information. Only use if the user explicitly asks to research,
    look up, or find external information. Never use for organizational topics.
    - sql: questions requiring live structured data (counts, sums, specific records).
    - visu: requests to visualize or chart data. Exact data from a prior specialist in this
    turn (e.g. sql) is passed through automatically, you do not need to repeat numbers in
    current_task, just describe what to chart.
    - convo: you can answer directly, no specialist needed (definitions, small talk, clarifying your
    own prior answer, or synthesizing/rephrasing when a plain specialist result isn't enough on its own).
    - clarification: the request could genuinely map to more than one route and cannot be
    disambiguated from the message alone.
    - end: the most recent message already fully answers the user's request and nothing further
    needs to happen. A completed specialist result (status="done") is presented to the user
    automatically when you choose end, you do not need to route to convo just to relay it.
 
    Reacting to the last specialist result:
    - status="done": the specialist succeeded. In almost all cases, choose next="end" directly,
    the result will be shown to the user automatically. Only route to convo instead if the raw
    result genuinely needs rephrasing, combining with another result, or the user asked something
    the specialist didn't fully address. Never route back to the same specialist that just succeeded.
    - Trust the status field, not the tone of the summary. A specialist may honestly note
    limitations, caveats, or missing details inside a status="done" summary, that is expected,
    honest behavior, not a sign of failure. Do not reinterpret a done result as incomplete,
    and do not route to clarification or back to the same specialist just because the summary
    hedges or mentions what it doesn't cover. status="done" means the specialist's job is
    finished; route to end.
    - status="partial" or "failed": read the issue field before deciding.
    - if issue is "permission_denied": this is a TERMINAL failure. No retry, rephrasing, or
        different specialist will change the outcome. Route to convo exactly once to explain the
        restriction, then end. Do not route back to sql or any specialist for this request again.
    - if the topic was organizational and RAG found nothing: respond via clarification or convo
        explaining no data exists. Do not fall through to research.
    - if the topic was non-organizational and nothing was found: research may be appropriate.
    - general rule: never route to the same specialist twice in a row for the same unresolved
        request, and never bounce back and forth between a specialist and convo more than once.
        If a specialist's failure isn't resolvable by retrying, explain the limitation via convo
        once, then end.
 
    For convo, also produce a short pre-summary of relevant conversation context in current_task,
    not a raw instruction, since convo will not see the full message history.
    For all other routes, current_task should be a concise actionable task description.
    For end, current_task can be a short, empty-ish placeholder, it will not be used.
 
    Never assume a request is restricted, confidential, or permission-denied on your own judgment.
    You do not know the user's actual permissions or what data exists. If the user asks for data,
    route to the appropriate specialist (sql, rag, etc.) and let it determine access and existence.
    Only treat something as a permission or access problem after last_result.issue reports it
    explicitly. Do not pre-emptively refuse a request based on what seems sensitive or confidential
    by general knowledge, that determination is not yours to make.
"""



def supervisor_agent(state: SupervisorState) -> dict:
    """
    Main Supervisor node. Gathers context, gets a validated routing
    decision from the LLM (with retry on malformed output), and maps
    it into the state update the graph applies.
    """

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

        print(
            "[SUPERVISOR] recorded task: "
            f"turn={current_turn} "
            f"route={task_record.route!r} "
            f"task={task_record.task!r} "
            f"status={task_record.status!r}"
        )

    # ---------------------------------------------------------
    # 2. Build context using the updated history
    # ---------------------------------------------------------

    context = gather_context(
        state=state,
        task_history=task_history,
    )

    model = llm()  

    try:
        decision = get_supervisor_decision(context, model)
    except Exception as e:

        raise RuntimeError(f"Supervisor LLM call failed: {e}") from e

    print(f"[SUPERVISOR] decision.next={decision.next!r} current_task={decision.current_task!r}")

    decision = enforce_task_history_guard(
        state=state,
        decision=decision,
        task_history=task_history,
    )

    update = map_to_state(decision)
    update["task_history"] = task_history

    return update

def get_current_turn(state: SupervisorState) -> int:
    """
    Derive the current user-turn number from the conversation.

    Specialist execution does not create new HumanMessages, so the number
    stays constant throughout one workflow execution.

    A genuinely new user request adds another HumanMessage and therefore
    gets a new turn number.
    """

    human_message_count = sum(
        1
        for message in state.messages
        if isinstance(message, HumanMessage)
    )

    return max(human_message_count, 1)

def gather_context(
    state: SupervisorState,
    task_history: list[TaskRecord],
) -> dict:
    print(
        f"[GATHER_CONTEXT] "
        f"{len(state.messages)} messages, "
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
        messages.append(SystemMessage(
            content=(
                f"Last specialist result — source: {lr.source}, "
                f"status: {lr.status}, issue: {lr.issue or 'none'}\n"
                f"Summary: {lr.summary}"
            )
        ))

    messages.extend(context["messages"])

    if previous_error:
        messages.append(SystemMessage(
            content=f"Your previous response failed validation: {previous_error}. "
                    f"Correct it and respond again in the required schema."
        ))

    return messages


def get_supervisor_decision(context: dict, model, max_attempts: int = 2) -> SupervisorDecision:
    """
    Calls the LLM for a routing decision, validating and retrying on
    malformed output. Falls back to clarification if it can't get a
    valid decision after max_attempts. Infra errors (not validation
    errors) are allowed to propagate, they're a different failure class.
    """
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        prompt = build_prompt(context, previous_error=last_error)

        try:
            raw_response = model.with_structured_output(SupervisorDecision, method="function_calling").invoke(prompt)
            return SupervisorDecision.model_validate(raw_response)

        except ValidationError as e:
            last_error = str(e)
            continue
        # deliberately not catching other exceptions here, e.g. network/rate-limit
        # errors should propagate to the node caller, not be treated as a
        # validation failure

    # exhausted retries on malformed output specifically
    return SupervisorDecision(
        next="clarification",
        current_task="The supervisor could not determine which capability should handle the request.",
    )

def enforce_task_history_guard(
    state: SupervisorState,
    decision: SupervisorDecision,
    task_history: list[TaskRecord],
) -> SupervisorDecision:
    """
    Deterministic protection against an exact repeat of a completed task
    within the same user turn.

    This does NOT prevent a specialist from running multiple times.
    It only prevents this pattern:

        visu("create chart X") -> done
        visu("create chart X") -> done
        visu("create chart X") -> ...

    A different task remains valid:

        research("find X") -> done
        research("find Y") -> done
    """

    if decision.next == "end":
        return decision

    if decision.next not in (
        "rag",
        "convo",
        "sql",
        "research",
        "visu",
    ):
        return decision

    current_turn = get_current_turn(state)

    proposed_task = (decision.current_task or "").strip()

    if not proposed_task:
        return decision

    for record in reversed(task_history):
        # Only compare work from this user's current turn.
        if record.turn != current_turn:
            continue

        # Failed/partial work may legitimately be retried.
        if record.status != "done":
            continue

        if record.route != decision.next:
            continue

        if record.task.strip() != proposed_task:
            continue

        # Exact same completed task requested again.
        print(
            "[SUPERVISOR GUARD] blocked duplicate completed task: "
            f"route={decision.next!r}, "
            f"task={proposed_task!r}"
        )

        return SupervisorDecision(
            next="end",
            current_task="",
        )

    return decision


def map_to_state(decision: SupervisorDecision) -> dict:
  
    if decision.next in ("end", "visu"):
        update = {
                "next": decision.next,
                "current_task": decision.current_task,
            }
    else:
        update = {
                "next": decision.next,
                "current_task": decision.current_task,
                "last_result": None,
            }
    return update