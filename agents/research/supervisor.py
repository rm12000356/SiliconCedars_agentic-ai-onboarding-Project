from langchain_core.messages import SystemMessage, HumanMessage
from state.state import SubGraphSupervisorState
from services.llm import llm
from pydantic import BaseModel, Field
from typing import Literal

class SubDecision(BaseModel):
    next: Literal["researcher", "report", "end"]
    reason: str = Field(description="Short explanation of the decision")

SUB_SUPERVISOR_PROMPT = """You are the controller of a research subgraph inside a larger multi-agent system.

Your only job is to decide the next step of the research process. You must choose exactly one of the following options:

- "researcher"  → More information still needs to be gathered from the web.
- "report"      → Enough information has been collected. It is time to write the final summary.
- "end"         → The research process is finished and should stop.

### Decision Rules (follow them strictly)

1. **No research done yet**
   - If there are no research findings in the messages yet → choose "researcher".

2. **Research is still incomplete**
   - If the researcher tried to gather information but the results are insufficient, incomplete, failed (e.g. 404, empty results, errors), or clearly need more work → choose "researcher".

3. **Enough information exists but no final report yet**
   - If the researcher has successfully gathered useful and relevant information, and a final summary/report has NOT been written yet → choose "report".

4. **Final report already exists**
   - If a clear final summary or report has already been produced (usually the last message is a polished answer with sources) → choose "end".
   - This is the most important rule to prevent infinite loops. Once a proper report exists, you must choose "end".

5. **Research completely failed**
   - If after multiple attempts the researcher found nothing useful, or clearly stated that no relevant information could be found → choose "end".

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

def Sub_controler(state: SubGraphSupervisorState) -> dict:
    model = llm().with_structured_output(SubDecision)

    messages = [
        SystemMessage(content=SUB_SUPERVISOR_PROMPT),
        HumanMessage(content=f"Original task: {state.task}"),
    ]

    if state.messages:
        # Give the controller a condensed view of what has been found so far
        last_content = state.messages[-1].content if state.messages else ""
        messages.append(HumanMessage(content=f"Latest research output:\n{last_content[:3000]}"))

    raw = model.invoke(messages)
    if isinstance(raw, SubDecision):
        decision = raw
    else:
        decision = SubDecision.model_validate(raw)

    print(f"[SUB-SUPERVISOR] next={decision.next} | reason={decision.reason}")

    return {"next": decision.next}