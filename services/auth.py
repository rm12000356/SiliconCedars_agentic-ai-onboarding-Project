from __future__ import annotations

from typing import Any, Optional

import bcrypt
from db.connection import get_elevated_connection


def _row_to_user(row: tuple) -> dict[str, Any]:
    # id, username, password_hash, permission_level, role, display_name, is_active
    return {
        "id": str(row[0]),
        "username": row[1],
        "password_hash": row[2],
        "permission_level": row[3],
        "role": row[4],
        "display_name": row[5] or row[1],
        "is_active": row[6],
    }


def get_user_by_username(username: str) -> Optional[dict[str, Any]]:
    """Return active app_users row as a dict, or None."""
    with get_elevated_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, username, password_hash, permission_level,
                       role, display_name, is_active
                FROM app_users
                WHERE username = %s AND is_active = TRUE
                """,
                (username,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return _row_to_user(row)


def verify_password(plain: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except Exception:
        return False


def authenticate(username: str, password: str) -> Optional[dict[str, Any]]:
    """
    Validate credentials against app_users.
    Returns user dict (without password_hash) on success, else None.
    """
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None

    # Never leak the hash to callers
    return {
        "id": user["id"],
        "username": user["username"],
        "permission_level": user["permission_level"],
        "role": user["role"],
        "display_name": user["display_name"],
    }