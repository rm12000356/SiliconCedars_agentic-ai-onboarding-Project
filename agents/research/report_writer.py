import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state.structure_output import ReportOutput
from state.state import SubGraphSupervisorState
from services.llm import llm
from services.budget import TurnBudgetExceeded
from services.message_utils import content_to_text

logger = logging.getLogger(__name__)

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


def Report_W(state: SubGraphSupervisorState) -> dict:
    logger.debug("[REPORT] task=%r", state.task)
    logger.debug("[REPORT] %s messages received", len(state.research_messages))
    if not state.research_messages:
        logger.debug("[REPORT] no messages at all, nothing to report on")
        return {
            "research_messages": [AIMessage(content="No research material was available.")],
            "research_succeeded": False,
            "report_written": True,
        }

    research_material = "\n\n".join(
        content_to_text(m.content) for m in state.research_messages
    )
    logger.debug("[REPORT] research_material=%r", research_material[:500])

    model = llm().with_structured_output(ReportOutput)
    try:
        result = model.invoke([
            SystemMessage(content=REPORT_PROMPT),
            HumanMessage(content=f"Original task: {state.task}\n\nResearch material:\n{research_material}")
        ])
    except TurnBudgetExceeded:
        raise
    # A report-writer outage must not crash the turn; return a structural
    # failure so the supervisor can explain and stop.
    except Exception as e:
        logger.warning(
            "[REPORT] generation failed: %s: %s", type(e).__name__, e
        )
        return {
            "research_messages": [
                AIMessage(content="The research report could not be generated.")
            ],
            "research_succeeded": False,
            "report_written": True,
        }

    if not isinstance(result, ReportOutput):
        result = ReportOutput.model_validate(result)

    logger.debug("[REPORT] success=%s final report: %r", result.success, result.content[:500])

    return {
        "research_messages": [AIMessage(content=result.content)],
        "research_succeeded": result.success,
        "report_written": True,
    }