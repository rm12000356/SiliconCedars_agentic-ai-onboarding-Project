from state.state import SupervisorState, SubGraphSupervisorState, SpecialistResult


def research_node(state: SupervisorState) -> dict:
    """
    Boundary wrapper for the Research subgraph.
    """
    if state.current_task is None:
        raise RuntimeError(
            "Convo node reached with current_task=None. The Supervisor "
            "should always set current_task before routing here."
        )

    sub_input = SubGraphSupervisorState(
        messages=[],
        task=state.current_task,
    )

    sub_output = _research_subgraph.invoke(sub_input)

    result = SpecialistResult(
        source="research",
        summary=sub_output["messages"][-1].content,
        status="done",
    )

    return {"last_result": result}


def set_research_subgraph(subgraph):
    global _research_subgraph
    _research_subgraph = subgraph