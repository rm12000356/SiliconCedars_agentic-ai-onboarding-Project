from typing import TypedDict, List, Annotated
from langgraph.graph.message import Literal, add_messages
from langchain_core.messages import AnyMessage

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


class SupervisorState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    next: list[MainRoute]

class SubGraphSupervisorState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    next: list[SubRoute]

class PLaceHolder(TypedDict):
    placeholder: str