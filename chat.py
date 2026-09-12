import os
from typing import Any, cast
import atexit
import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from langchain_core.runnables import RunnableConfig
from services.memory import get_checkpointer

from graph.workflow import Main_WorkFlow


load_dotenv()


@cl.data_layer
def get_data_layer():
    database_url = os.getenv("CHAINLIT_DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "CHAINLIT_DATABASE_URL is not set. "
            "Please add it to your .env file."
        )

    return SQLAlchemyDataLayer(conninfo=database_url)


memory_context = get_checkpointer()
memory = memory_context.__enter__()

graph = Main_WorkFlow(memory)

print("GRAPH:", type(graph))
print(
    "CHECKPOINTER:",
    getattr(graph, "checkpointer", "NO CHECKPOINTER ATTRIBUTE")
)

def close_memory(
    typ: type[BaseException] | None,
    value: BaseException | None,
    traceback: Any | None,
) -> bool | None:
    return memory_context.__exit__(typ, value, traceback)


atexit.register(close_memory, None, None, None)

@cl.on_chat_start
async def start():
    # This tracks whether the LangGraph thread is currently
    # waiting for an answer to an interrupt.
    cl.user_session.set("awaiting_clarification", False)


@cl.on_message
async def main(message: cl.Message):

    thread_id = cl.context.session.thread_id
    user_id = "test-user-1"
    permission_level = os.getenv("DEFAULT_PERMISSION_LEVEL", "general")

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "permission_level": permission_level,
        }
    }

    awaiting_clarification = cl.user_session.get(
        "awaiting_clarification"
    )

    # ---------------------------------------------------------
    # RESUME A PAUSED GRAPH
    # ---------------------------------------------------------
    if awaiting_clarification:

        result = graph.invoke(
            Command(resume=message.content),
            config=config,
        )

        # The clarification has now been answered.
        cl.user_session.set("awaiting_clarification", False)

    # ---------------------------------------------------------
    # START A NEW GRAPH INVOCATION
    # ---------------------------------------------------------
    else:

        result = graph.invoke(
            cast(
                Any,
                {
                    "messages": [
                        HumanMessage(content=message.content)
                    ]
                },
            ),
            config=config,
        )

    # ---------------------------------------------------------
    # CHECK IF LANGGRAPH PAUSED AT AN INTERRUPT
    # ---------------------------------------------------------
    interrupts = result.get("__interrupt__")

    if interrupts:

        # The graph has paused and is waiting for the user.
        cl.user_session.set("awaiting_clarification", True)

        interrupt_value = interrupts[0].value

        if isinstance(interrupt_value, dict):
            question = interrupt_value.get(
                "question",
                "Could you clarify your request?",
            )
        else:
            question = str(interrupt_value)

        await cl.Message(content=question).send()

        return

    # ---------------------------------------------------------
    # NORMAL GRAPH COMPLETION / CONTINUATION
    # ---------------------------------------------------------
    messages = result.get("messages", [])

    if messages:
        response = messages[-1].content
        await cl.Message(content=str(response)).send()