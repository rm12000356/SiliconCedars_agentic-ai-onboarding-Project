"""Unit tests for services.auth — no database, no network.

`get_user_by_username` is the only DB-touching function; it is monkeypatched so
the bcrypt verification and the "never leak the hash" contract can be exercised
without Postgres.
"""

from __future__ import annotations

import bcrypt

from services import auth


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()


def _user(password: str = "correct-horse") -> dict:
    return {
        "id": "11111111-1111-1111-1111-111111111111",
        "username": "alice",
        "password_hash": _hash(password),
        "permission_level": "elevated",
        "role": "manager",
        "display_name": "Alice",
        "is_active": True,
    }


def test_authenticate_success_returns_public_fields_without_hash(monkeypatch):
    monkeypatch.setattr(auth, "get_user_by_username", lambda _u: _user())

    result = auth.authenticate("alice", "correct-horse")

    assert result is not None
    assert result["username"] == "alice"
    assert result["permission_level"] == "elevated"
    assert result["role"] == "manager"
    assert result["display_name"] == "Alice"
    assert "password_hash" not in result


def test_authenticate_wrong_password_returns_none(monkeypatch):
    monkeypatch.setattr(auth, "get_user_by_username", lambda _u: _user())
    assert auth.authenticate("alice", "wrong") is None


def test_authenticate_unknown_user_returns_none(monkeypatch):
    monkeypatch.setattr(auth, "get_user_by_username", lambda _u: None)
    assert auth.authenticate("ghost", "whatever") is None


def test_authenticate_unknown_user_still_runs_dummy_check(monkeypatch):
    """A missing user must still pay the bcrypt cost, so timing does not
    reveal which usernames exist."""
    calls: list[tuple[str, str]] = []

    def _record(plain: str, hashed: str) -> bool:
        calls.append((plain, hashed))
        return False

    monkeypatch.setattr(auth, "get_user_by_username", lambda _u: None)
    monkeypatch.setattr(auth, "verify_password", _record)

    assert auth.authenticate("ghost", "whatever") is None
    assert calls == [("whatever", auth._DUMMY_PASSWORD_HASH)]


def test_verify_password_accepts_matching_hash():
    assert auth.verify_password("secret", _hash("secret")) is True


def test_verify_password_rejects_mismatch():
    assert auth.verify_password("secret", _hash("other")) is False


def test_verify_password_invalid_hash_is_false_not_raises():
    assert auth.verify_password("secret", "not-a-bcrypt-hash") is False


def test_row_to_user_falls_back_to_username_for_display_name():
    row = (
        "22222222-2222-2222-2222-222222222222",
        "bob",
        "$2b$12$hash",
        "general",
        "analyst",
        None,
        True,
    )

    user = auth._row_to_user(row)

    assert user["id"] == "22222222-2222-2222-2222-222222222222"
    assert user["display_name"] == "bob"
    assert user["permission_level"] == "general"
