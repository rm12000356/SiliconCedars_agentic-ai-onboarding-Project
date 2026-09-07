from state.state import SupervisorState
from state.structure_output import SupervisorDecision
from pydantic import ValidationError
from langchain_core.messages import SystemMessage
from services.llm import llm

SUPERVISOR_SYSTEM_PROMPT = """You are the routing supervisor for a company intelligence assistant.
    Based on the conversation, decide which specialist should handle the latest request.

    Routes:
    - rag: internal company documents, policies, procedures. DEFAULT for organizational questions.
    - research: external/public information. Only use if the user explicitly asks to research,
    look up, or find external information. Never use for organizational topics.
    - sql: questions requiring live structured data (counts, sums, specific records).
    - visu: requests to visualize or chart data.
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

    model = llm()  
    
    context = gather_context(state)

    try:
        decision = get_supervisor_decision(context, model)
    except Exception as e:
        # infra-level failure (network, rate limit, provider outage),
        # distinct from a validation failure, let it propagate rather
        # than silently defaulting to clarification
        raise RuntimeError(f"Supervisor LLM call failed: {e}") from e

    print(f"[SUPERVISOR] decision.next={decision.next!r} current_task={decision.current_task!r}")

    return map_to_state(decision)



def gather_context(state: SupervisorState) -> dict:
    """
    Pulls the minimum needed from state to build the routing prompt.
    Doesn't touch messages directly here beyond what's needed;
    the Supervisor gets full history, but we're explicit about
    what's passed forward so build_prompt isn't reaching into state itself.
    """
    print(f"[GATHER_CONTEXT] {len(state.messages)} messages, last_result={state.last_result!r}")
    return {
        "messages": state.messages,
        "last_result": state.last_result,  # None if this is the first turn
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
            raw_response = model.with_structured_output(SupervisorDecision).invoke(prompt)
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

def map_to_state(decision: SupervisorDecision) -> dict:
    
    if decision.next != "end":
        update = {
                "next": decision.next,
                "current_task": decision.current_task,
                "last_result": None,  
            }
    else:
        update = {
                "next": decision.next,
                "current_task": decision.current_task,
            }
    return update
