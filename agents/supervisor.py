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
- convo: you can answer directly, no specialist needed (definitions, small talk, clarifying your own prior answer).
- clarification: the request could genuinely map to more than one route and cannot be
  disambiguated from the message alone.
- end: the most recent message already fully answers the user's request and nothing
  further needs to happen. Use this whenever the last message in the conversation is
  an assistant response (from convo or a specialist) that already satisfies what the
  user asked for. Do not route back to convo or any specialist a second time for the
  same completed request, that just repeats the same answer.
 
If the last specialist result has status "partial" or "failed", factor that in:
- if the topic was organizational and RAG found nothing, respond via clarification or convo
  explaining no data exists. Do not fall through to research.
- if the topic was non-organizational and nothing was found, research may be appropriate.
 
For convo, also produce a short pre-summary of relevant conversation context in current_task,
not a raw instruction, since convo will not see the full message history.
For all other routes, current_task should be a concise actionable task description.
For end, current_task can be a short, empty-ish placeholder, it will not be used.
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
            content=f"Last specialist result — source: {lr.source}, "
                    f"status: {lr.status}, issue: {lr.issue or 'none'}"
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
    return {
        "next": decision.next,
        "current_task": decision.current_task,
        "last_result": None, 
    }
