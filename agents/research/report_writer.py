from langchain_core.messages import SystemMessage, HumanMessage
from state.state import SubGraphSupervisorState
from services.llm import llm

REPORT_PROMPT = """You are the final report writer for a research task.

Using only the research material provided, write a short, clear summary that answers the original task.
Always include the sources you used (title + url).

If the research material indicates that nothing relevant was found, clearly say so.

Keep the answer concise and factual.
"""


def Report_W(state: SubGraphSupervisorState) -> dict:
    print(f"[REPORT] task={state.task!r}")
    print(f"[REPORT] {len(state.messages)} messages received")
    if not state.messages:
        print("[REPORT] no messages at all, nothing to report on")
        content = "No research material was available."
    else:
        # Use the accumulated research messages
        research_material = "\n\n".join(
            _content_to_str(m.content) for m in state.messages
        )
        print(f"[REPORT] research_material={research_material[:500]!r}")
        model = llm()
        response = model.invoke([
            SystemMessage(content=REPORT_PROMPT),
            HumanMessage(content=f"Original task: {state.task}\n\nResearch material:\n{research_material}")
        ])
        content = response.content
        print(f"[REPORT] final report: {str(content)[:500]!r}")
        
    return {"messages": [HumanMessage(content=content)]}


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