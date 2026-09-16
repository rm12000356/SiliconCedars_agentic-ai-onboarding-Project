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

        succeeded = sub_output.get("research_succeeded")
        research_messages = sub_output.get("research_messages") or []

        if not research_messages:
            return {
                "last_result": SpecialistResult(
                    source="research",
                    summary=(
                        "The research step finished without producing any "
                        "material to report."
                    ),
                    status="failed",
                    issue="research_no_results",
                )
            }

        summary = str(research_messages[-1].content).strip()
        if not summary:
            return {
                "last_result": SpecialistResult(
                    source="research",
                    summary="The research step produced an empty report.",
                    status="failed",
                    issue="research_empty_report",
                )
            }

        result = SpecialistResult(
            source="research",
            summary=summary,
            status="done" if succeeded else "failed",
            issue=None if succeeded else "research_no_results",
        )

        return {"last_result": result}

    return research_node