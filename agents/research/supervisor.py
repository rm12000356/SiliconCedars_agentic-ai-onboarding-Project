from state.state import SubGraphSupervisorState

def Sub_controler(state: SubGraphSupervisorState) -> dict:
    print("Sub_controler called with task:", state.task)
    return {"next": "end"}  # placeholder: immediately ends the subgraph