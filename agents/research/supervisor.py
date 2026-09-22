import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state.state import SubGraphSupervisorState, SubDecision
from services.llm import llm
from services.budget import TurnBudgetExceeded
from services.message_utils import content_to_text

logger = logging.getLogger(__name__)

SUB_SUPERVISOR_PROMPT = """You are the controller of a research subgraph inside a larger multi-agent system.

Your only job is to decide the next step of the research process. You must choose exactly one of the following options:

- "researcher"  → More information still needs to be gathered from the web.
- "report"      → Enough information has been collected. It is time to write the final summary.
- "end"         → The research process is finished and should stop.

### Decision Rules (follow them strictly)

1. **No research done yet**
   - If there are no research findings in the messages yet → choose "researcher".

2. **Prefer finishing over perfection**
   - If the latest research output already contains concrete facts, dates, tables, multiple sources, or a structured note, choose "report" even if the note feels slightly incomplete. Do not keep researching for perfection.

3. **Research is still incomplete**
   - Only choose "researcher" again when the results are truly insufficient, empty, failed (404, empty body, errors), or clearly missing the core of the original task.

4. **Enough information exists but no final report yet**
   - If the researcher has successfully gathered useful and relevant information, and a final summary/report has NOT been written yet → choose "report".

5. **Final report already exists**
   - If a clear final summary or report has already been produced → choose "end".
   - This is the most important rule to prevent infinite loops.

6. **Research completely failed**
   - If after multiple attempts the researcher found nothing useful → choose "end".

### Additional Guidelines

- Prefer moving forward rather than looping.
- Never choose "report" if a final report has already been written.
- Never choose "researcher" if solid information already exists and only the final write-up is missing.
- Be conservative: only send back to "researcher" when more information is genuinely needed.
- Your decision must be based on the actual content of the messages, not on assumptions.

### Output
You must respond with a structured decision containing:
- next: one of "researcher", "report", or "end"
- reason: a short and clear explanation of why you made this choice
"""

MAX_RESEARCH_ATTEMPTS = 3

def _has_substantial_note(messages) -> bool:
    """Cheap heuristic: any message longer than ~800 chars is treated as real material."""
    for m in messages:
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):          # multimodal safety
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        if len(str(content)) > 800:
            return True
    return False

def _clean_latest_content(messages, max_chars: int = 2000) -> str:
    """
    Give the controller a cleaner view:
    - prefer the last AIMessage (the researcher's synthesis)
    - fall back to the last message
    - strip tool-call noise and truncate
    """
    if not messages:
        return ""

    # Prefer the last AI message (usually the research note)
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            text = content_to_text(m.content)
            break
    else:
        text = content_to_text(messages[-1].content) if messages else ""

    # Light cleanup of obvious tool-call boilerplate
    for noise in ("tool call:", "tool result:", "args=", "FunctionMessage"):
        text = text.replace(noise, "")

    return text[:max_chars]


def sub_controller(state: SubGraphSupervisorState) -> dict:

    if state.report_written:
        logger.debug("[SUB-SUPERVISOR] report already written, ending research subgraph")
        return {"next": "end"}

    if state.research_attempts >= MAX_RESEARCH_ATTEMPTS:
        logger.debug(
            "[SUB-SUPERVISOR] research_attempts=%s >= %s, forcing report",
            state.research_attempts, MAX_RESEARCH_ATTEMPTS,
        )
        return {"next": "report"}

    if state.research_attempts >= 1 and _has_substantial_note(state.research_messages):
        logger.debug("[SUB-SUPERVISOR] substantial research note already present -> forcing report")
        return {"next": "report"}

    # Nothing has been researched yet: the answer is known without the LLM.
    if state.research_attempts == 0 and not state.research_messages:
        logger.debug("[SUB-SUPERVISOR] no research yet -> researcher (no LLM call)")
        return {"next": "researcher", "research_attempts": 1}

    model = llm().with_structured_output(SubDecision)

    messages = [
        SystemMessage(content=SUB_SUPERVISOR_PROMPT),
        HumanMessage(content=f"Original task: {state.task}"),
    ]

    latest = _clean_latest_content(state.research_messages)
    if latest:
        messages.append(HumanMessage(content=f"Latest research output:\n{latest}"))

    try:
        raw = model.invoke(messages)
    except TurnBudgetExceeded:
        raise
    # A controller outage should not crash the whole turn: move to the report
    # step, which will itself degrade if the provider is still down.
    except Exception as e:
        logger.warning(
            "[SUB-SUPERVISOR] decision failed, forcing report: %s: %s",
            type(e).__name__, e,
        )
        return {"next": "report"}

    if isinstance(raw, SubDecision):
        decision = raw
    else:
        decision = SubDecision.model_validate(raw)

    logger.debug("[SUB-SUPERVISOR] next=%s | reason=%s", decision.next, decision.reason)

    update: dict[str, object] = {"next": decision.next}

    if decision.next == "researcher":
        update["research_attempts"] = state.research_attempts + 1

    return update