from state.state import SpecialistResult

from state.state import SupervisorState, SpecialistResult

def Visualization(state: SupervisorState) -> dict:
    print("Visualization called with task:", state.current_task)
    result = SpecialistResult(
        source="visu",
        summary="Placeholder Visualization response.",
        status="done",
    )
    return {"last_result": result}