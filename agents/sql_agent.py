from state.state import SupervisorState, SpecialistResult

def Sql_agent(state: SupervisorState) -> dict:
    print("Sql_agent called with task:", state.current_task)
    result = SpecialistResult(
        source="sql",
        summary="Placeholder SQL response.",
        status="done",
    )
    return {"last_result": result}