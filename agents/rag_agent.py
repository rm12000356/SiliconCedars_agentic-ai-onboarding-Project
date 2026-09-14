import logging
import os

from langchain_core.messages import SystemMessage, HumanMessage

from state.state import SupervisorState, SpecialistResult
from services.llm import llm
from rag.retrieval import retrieve_relevant_chunks

logger = logging.getLogger("rag")

NO_MATCH_DISTANCE_THRESHOLD = float(os.getenv("RAG_NO_MATCH_DISTANCE_THRESHOLD", "0.8"))


def RAG(state: SupervisorState) -> dict:
    """
    Retrieves relevant internal documents and generates a grounded answer
    using only that retrieved content. If nothing relevant is found, reports
    status="partial" rather than guessing or falling back to outside
    knowledge 
    """
    if state.current_task is None:
        raise RuntimeError(
            "RAG node reached with current_task=None. The Supervisor "
            "should always set current_task before routing here."
        )

    try:
        chunks = retrieve_relevant_chunks(state.current_task, top_k=3)
    except Exception as e:
        logger.warning(
            "retrieval_failed",
            extra={"error_type": type(e).__name__, "error": str(e)},
        )
        result = SpecialistResult(
            source="rag",
            summary="The internal document search is temporarily unavailable.",
            status="failed",
            issue="rag_unavailable",
        )
        return {"last_result": result}

    if chunks:
        distances = [c["distance"] for c in chunks]
        logger.debug(
            "retrieval_distances",
            extra={
                "min": min(distances),
                "max": max(distances),
                "n_below_threshold": sum(d < NO_MATCH_DISTANCE_THRESHOLD for d in distances),
                "n_total": len(distances),
                "threshold": NO_MATCH_DISTANCE_THRESHOLD,
            },
        )
    else:
        logger.debug("retrieval_returned_no_chunks")

    relevant_chunks = [c for c in chunks if c["distance"] < NO_MATCH_DISTANCE_THRESHOLD]

    if not relevant_chunks:
        result = SpecialistResult(
            source="rag",
            summary="No matching internal documents were found for this request.",
            status="partial",
            issue="no_matching_documents",
        )
        return {"last_result": result}

    context_text = "\n\n".join(f"- {c['content']}" for c in relevant_chunks)
    system_message = SystemMessage(
        content=(
            "You are a RAG specialist answering questions using only the internal "
            "company documents provided below. Do not use outside knowledge. If the "
            "provided documents don't fully answer the question, say what's missing "
            "rather than guessing.\n\n"
            f"Retrieved documents:\n{context_text}"
        )
    )

    model = llm()
    response = model.invoke([system_message, HumanMessage(content=state.current_task)])

    result = SpecialistResult(
        source="rag",
        summary=str(response.content),
        status="done",
    )
    return {"last_result": result}