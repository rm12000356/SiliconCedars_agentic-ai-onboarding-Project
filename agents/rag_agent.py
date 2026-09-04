from state.state import SupervisorState, SpecialistResult

def RAG(state: SupervisorState) -> dict:
    print("RAG called with task:", state.current_task)
    result = SpecialistResult(
        source="rag",
        summary="Placeholder RAG response.",
        status="done",
    )
    return {"last_result": result}