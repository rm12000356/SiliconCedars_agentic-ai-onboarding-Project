from __future__ import annotations

import os
import threading

from datetime import datetime, timezone
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from db.connection import get_elevated_connection
from dotenv import load_dotenv

from state.state import SpecialistResult, TaskRecord

load_dotenv()

_CHECKPOINT_ALLOWED_TYPES = [TaskRecord, SpecialistResult]

_CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=_CHECKPOINT_ALLOWED_TYPES,
)


def get_checkpointer():
    backend = os.getenv("CHECKPOINT_BACKEND", "memory").lower()

    if backend == "postgres":
        conn_string = os.getenv("DATABASE_URL")
        if not conn_string:
            raise ValueError("DATABASE_URL must be set when CHECKPOINT_BACKEND is postgres")

        saver = PostgresSaver.from_conn_string(conn_string)
        checkpointer = saver.__enter__()
        checkpointer.setup()          # creates the checkpoint tables if needed
        checkpointer.serde = _CHECKPOINT_SERDE
        return checkpointer , saver

    return MemorySaver(serde=_CHECKPOINT_SERDE), None



_memory_table_lock = threading.Lock()
_memory_table_ready = False


def _ensure_memory_table() -> None:
    """Create the long-term memory table once per process (avoids a DDL
    round trip on every long-term-memory operation)."""
    global _memory_table_ready
    if _memory_table_ready:
        return

    with _memory_table_lock:
        if _memory_table_ready:
            return
        with get_elevated_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_memory (
                        user_id     TEXT NOT NULL,
                        key         TEXT NOT NULL,
                        value       TEXT NOT NULL,
                        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (user_id, key)
                    );
                """)
            conn.commit()
        _memory_table_ready = True


def write_fact(user_id: str, key: str, value: str) -> None:
    """Store or update a long-term fact for a user."""
    _ensure_memory_table()
    with get_elevated_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_memory (user_id, key, value, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                """,
                (user_id, key, value, datetime.now(timezone.utc)),
            )
        conn.commit()


def read_facts(user_id: str) -> dict[str, str]:
    """Return all long-term facts for a user as {key: value}."""
    _ensure_memory_table()
    with get_elevated_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT key, value FROM user_memory WHERE user_id = %s",
                (user_id,),
            )
            rows = cur.fetchall()
    return {row[0]: row[1] for row in rows}


def format_facts_for_prompt(user_id: str) -> str:
    """
    Returns a short block ready to inject into a system prompt.
    Empty string if the user has no facts yet.
    """
    facts = read_facts(user_id)
    if not facts:
        return ""
    lines = [f"- {k}: {v}" for k, v in facts.items()]
    return "Known facts about this user:\n" + "\n".join(lines)


def clear_user_memory(user_id: str) -> None:
    """Debug helper – wipe all long-term memory for a user."""
    _ensure_memory_table()
    with get_elevated_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_memory WHERE user_id = %s", (user_id,))
        conn.commit()