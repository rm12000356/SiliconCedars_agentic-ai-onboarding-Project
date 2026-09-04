from state.state import SupervisorState, SubGraphSupervisorState, SpecialistResult


def make_research_node(subgraph):
    """
    Factory that builds the research_node function with the compiled
    Research subgraph closed over directly so it doesn't need to be passed in every time.
    """

    def research_node(state: SupervisorState) -> dict:
        """
        Boundary wrapper for the Research subgraph. Invokes the subgraph
    
        """
        if state.current_task is None:
            raise RuntimeError(
                "Research node reached with current_task=None. The Supervisor "
                "should always set current_task before routing here."
            )

        sub_input = SubGraphSupervisorState(
            messages=[],
            task=state.current_task,
        )

        sub_output = subgraph.invoke(sub_input)

        result = SpecialistResult(
            source="research",
            summary=sub_output["messages"][-1].content,
            status="done",
        )

        return {"last_result": result}

    return research_node