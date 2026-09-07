from langchain_core.messages import SystemMessage, HumanMessage
from state.state import SupervisorState, SpecialistResult
from services.llm import llm
from rag.retrieval import retrieve_relevant_chunks

# Cosine distance threshold below which a chunk is considered a real
# match. Starting point, not tuned yet, lower = more similar, 0 = identical.
# Worth adjusting once you see real retrieval results, this number is a
# guess until tested against actual queries.
NO_MATCH_DISTANCE_THRESHOLD = 0.5


def RAG(state: SupervisorState) -> dict:
    """
    Retrieves relevant internal documents and generates a grounded
    answer using only that retrieved content. If nothing relevant is
    found, reports status="partial" rather than guessing or falling
    back to outside knowledge, this is what lets the Supervisor's
    organizational-fallback rule (no fallthrough to research for
    internal topics) actually function.
    """
    if state.current_task is None:
        raise RuntimeError(
            "RAG node reached with current_task=None. The Supervisor "
            "should always set current_task before routing here."
        )

    chunks = retrieve_relevant_chunks(state.current_task, top_k=3)
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