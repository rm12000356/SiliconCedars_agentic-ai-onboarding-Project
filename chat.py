import os
from typing import Any, Optional, cast
import atexit
import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from langchain_core.runnables import RunnableConfig
from services.memory import get_checkpointer
from graph.workflow import Main_WorkFlow
from services.auth import authenticate

load_dotenv()

@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> Optional[cl.User]:
    user = authenticate(username, password)
    if not user:
        return None

    return cl.User(
        identifier=user["username"],  # stable user_id for long-term memory
        display_name=user["display_name"],
        metadata={
            "user_uuid": user["id"],
            "permission_level": user["permission_level"],
            "role": user["role"],
            "provider": "credentials",
        },
    )

@cl.data_layer
def get_data_layer():
    database_url = os.getenv("CHAINLIT_DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "CHAINLIT_DATABASE_URL is not set. "
            "Please add it to your .env file."
        )
    return SQLAlchemyDataLayer(conninfo=database_url)


memory, memory_context = get_checkpointer()
graph = Main_WorkFlow(memory)

print("GRAPH:", type(graph))
print("CHECKPOINTER:", getattr(graph, "checkpointer", "NO CHECKPOINTER ATTRIBUTE"))


def close_memory(
    typ: type[BaseException] | None,
    value: BaseException | None,
    traceback: Any | None,
) -> bool | None:
    if memory_context is not None:
        return memory_context.__exit__(typ, value, traceback)
    return None


atexit.register(close_memory, None, None, None)


@cl.on_chat_start
async def start():
    cl.user_session.set("awaiting_clarification", False)

    app_user = cl.user_session.get("user")
    if app_user:
        role = (app_user.metadata or {}).get("role", "user")
        await cl.Message(
            content=f"Welcome, **{app_user.identifier}** ({role}). How can I help you today?"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    app_user = cl.user_session.get("user")
    if not app_user:
        await cl.Message(content="Please log in first.").send()
        return

    # Real identity from login
    user_id = app_user.identifier
    permission_level = (app_user.metadata or {}).get(
        "permission_level",
        os.getenv("DEFAULT_PERMISSION_LEVEL", "general"),
    )

    thread_id = cl.context.session.thread_id

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "permission_level": permission_level,
        }
    }

    awaiting_clarification = cl.user_session.get("awaiting_clarification")

    # ---------------------------------------------------------
    # RESUME A PAUSED GRAPH
    # ---------------------------------------------------------
    if awaiting_clarification:
        result = graph.invoke(
            Command(resume=message.content),
            config=config,
        )
        cl.user_session.set("awaiting_clarification", False)


    else:
        result = graph.invoke(
            cast(
                Any,
                {
                    "messages": [HumanMessage(content=message.content)]
                },
            ),
            config=config,
        )

    interrupts = result.get("__interrupt__")

    if interrupts:
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

    messages = result.get("messages", [])
    if messages:
        response = messages[-1].content
        await cl.Message(content=str(response)).send()