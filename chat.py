import asyncio
import logging
import os
from typing import Any, Optional, cast
import atexit
import psycopg
from psycopg_pool import PoolTimeout
import chainlit as cl
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from chainlit.auth import get_current_user
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.server import app
from langchain_core.runnables import RunnableConfig
from services.memory import get_checkpointer
from services.chart_storage import LocalChartStorage
from graph.workflow import Main_WorkFlow, has_pending_interrupt
from services.auth import authenticate
from services.budget import budget_scope, TurnBudgetExceeded
from services.logging_config import configure_logging
from agents.finalize import OUTAGE_MESSAGE

load_dotenv()
configure_logging()

logger = logging.getLogger(__name__)

CHART_STORAGE = LocalChartStorage()

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
    return SQLAlchemyDataLayer(conninfo=database_url, storage_provider=CHART_STORAGE)


chart_routes = APIRouter(dependency_overrides_provider=app)


@chart_routes.get("/charts/{token}")
async def serve_chart(token: str, user=Depends(get_current_user)):
    """
    Authenticated, owner-scoped endpoint for persisted chart images.

    LocalChartStorage.get_read_url returns an opaque base64url token.
    get_current_user enforces login, and the token's object key is bound to
    the requester's user id so one user cannot fetch another user's chart.
    """
    try:
        object_key = CHART_STORAGE.decode_token(token)
        path = CHART_STORAGE.resolve(object_key)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=404, detail="Chart not found")

    current_id = getattr(user, "id", None)
    if not current_id or CHART_STORAGE.owner_of(object_key) != str(current_id):
        raise HTTPException(status_code=404, detail="Chart not found")

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Chart not found")

    return FileResponse(path, media_type="image/png")


# Chainlit registers an SPA catch-all ("/{full_path:path}") inside the router it
# includes at import time, so a route added to `app` afterwards is shadowed and
# every /charts request returns index.html. Splice ours in ahead of the included
# router so FastAPI matches it first.
_included_router_idx = next(
    (
        index
        for index, route in enumerate(app.router.routes)
        if type(route).__name__ == "_IncludedRouter"
    ),
    0,
)
app.router.routes[_included_router_idx:_included_router_idx] = chart_routes.routes


memory, memory_context = get_checkpointer()
graph = Main_WorkFlow(memory)

logger.info("GRAPH: %s", type(graph))
logger.info(
    "CHECKPOINTER: %s",
    getattr(graph, "checkpointer", "NO CHECKPOINTER ATTRIBUTE"),
)


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
    app_user = cl.user_session.get("user")
    if app_user:
        role = (app_user.metadata or {}).get("role", "user")
        await cl.Message(
            content=f"Welcome, **{app_user.identifier}** ({role}). How can I help you today?"
        ).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    """
    Chainlit only resumes a persisted thread when this hook is registered.
    Without it a browser refresh leaves the UI on the thread preview with no
    composer. Chainlit replays persisted messages/elements and restores the
    user session automatically; nothing else is required here. Do not send
    messages in this hook (known Chainlit issues #2338 / #2611 make them
    vanish). The LangGraph graph is keyed by cl.context.session.thread_id,
    which is the resumed thread's id, so its checkpoint is reused.
    """
    pass


@cl.on_message
async def main(message: cl.Message):
    app_user = cl.user_session.get("user")
    if not app_user:
        await cl.Message(content="Please log in first.").send()
        return

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

    try:
        snapshot = await asyncio.to_thread(graph.get_state, config)
        paused = has_pending_interrupt(snapshot)
    except (RuntimeError, ValueError, psycopg.Error, PoolTimeout):
        logger.warning(
            "get_state failed; assuming no pending interrupt", exc_info=True
        )
        paused = False

    # One aggregate LLM budget per inbound message. A clarification resume
    # arrives as a new on_message call, so Chainlit budgets are per-message,
    # not per logical turn (unlike the CLI, which spans the clarification loop).
    try:
        with budget_scope():
            if paused:
                result = await asyncio.to_thread(
                    graph.invoke, Command(resume=message.content), config
                )
            else:
                result = await asyncio.to_thread(
                    graph.invoke,
                    cast(Any, {"messages": [HumanMessage(content=message.content)]}),
                    config,
                )
    except (TurnBudgetExceeded, RuntimeError):
        logger.exception("graph invoke failed; returning outage message")
        await cl.Message(content=OUTAGE_MESSAGE).send()
        return

    interrupts = result.get("__interrupt__")

    if interrupts:
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

        elements = []
        chart_path = result.get("chart_path")
        if chart_path and os.path.exists(chart_path):
            elements.append(
                cl.Image(
                    path=chart_path,
                    name=os.path.basename(chart_path),
                    display="inline",
                )
            )

        await cl.Message(content=str(response), elements=elements).send()