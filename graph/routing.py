from typing import Literal
from state.state import SupervisorState, SubGraphSupervisorState

MainRoute = Literal[
    "rag",
    "convo",
    "sql",
    "visu",
    "worker",
    "end",
]

SubRoute = Literal[
    "research",
    "report",
    "end",
]


def route_supervisor(state: SupervisorState)->  list[MainRoute]:
    return state["next"]


def route_sub_supervisor(state: SubGraphSupervisorState)->  list[SubRoute]:
    return state["next"]