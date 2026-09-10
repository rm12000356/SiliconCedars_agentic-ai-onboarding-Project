from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from state.state import SubGraphSupervisorState
from services.llm import llm

REPORT_PROMPT = """You are the final report writer for a research task.

Using only the research material provided, write a short, clear summary that answers the original task.
Always include the sources you used (title + url).

If the research material indicates that nothing relevant was found, clearly say so.

Keep the answer concise and factual.

You must also indicate, as a separate structured field, whether the research material
actually contains genuine information that answers the task. This is a factual
determination, not about how the report is phrased: did real, usable information get
found, or not. Do not let politeness or hedging affect this field, if nothing useful
was found, say so plainly here even if the report text itself is written diplomatically.
"""


class ReportOutput(BaseModel):
    content: str = Field(description="The final report text to show the user.")
    success: bool = Field(
        description="True only if genuine information answering the task was found. "
                    "False if the research material shows repeated failures, blocked "
                    "sources, or no usable data, regardless of how the report is worded."
    )


def Report_W(state: SubGraphSupervisorState) -> dict:
    print(f"[REPORT] task={state.task!r}")
    print(f"[REPORT] {len(state.messages)} messages received")
    if not state.messages:
        print("[REPORT] no messages at all, nothing to report on")
        return {
            "messages": [HumanMessage(content="No research material was available.")],
            "research_succeeded": False,
            "report_written": True,
        }

    # Use the accumulated research messages
    research_material = "\n\n".join(
        _content_to_str(m.content) for m in state.messages
    )
    print(f"[REPORT] research_material={research_material[:500]!r}")

    model = llm().with_structured_output(ReportOutput)
    result = model.invoke([
        SystemMessage(content=REPORT_PROMPT),
        HumanMessage(content=f"Original task: {state.task}\n\nResearch material:\n{research_material}")
    ])
    if not isinstance(result, ReportOutput):
        result = ReportOutput.model_validate(result)

    print(f"[REPORT] success={result.success} final report: {result.content[:500]!r}")

    return {
        "messages": [HumanMessage(content=result.content)],
        "research_succeeded": result.success,
        "report_written": True,
    }


def _content_to_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)