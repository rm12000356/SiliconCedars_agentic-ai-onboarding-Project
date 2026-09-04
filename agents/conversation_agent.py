from langchain_core.messages import AIMessage
from state.state import SupervisorState

def Convo(state: SupervisorState) -> dict:
    return {"messages": [AIMessage(content=f"[stub convo response to: {state.current_task}]")]}
