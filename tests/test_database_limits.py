from __future__ import annotations

import pytest

from tools.database import MAX_RESULT_BYTES, MAX_ROWS, run_general_query


class _FakeCursor:
    def __init__(self, description, rows):
        self.description = description
        self._rows = rows
        self.executed: list[str] = []

    def execute(self, query, params=None):
        self.executed.append(str(query))

    def fetchmany(self, n):
        return self._rows[:n]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch(monkeypatch, cursor):
    monkeypatch.setattr(
        "tools.database.get_general_connection", lambda: _FakeConn(cursor)
    )


def test_query_runs_in_read_only_transaction(monkeypatch):
    cursor = _FakeCursor([("n",)], [(1,)])
    _patch(monkeypatch, cursor)

    run_general_query.invoke({"query_text": "SELECT 1"})

    assert "READ ONLY" in cursor.executed[0].upper()


def test_oversized_row_count_is_rejected(monkeypatch):
    rows = [(i,) for i in range(MAX_ROWS + 1)]
    cursor = _FakeCursor([("n",)], rows)
    _patch(monkeypatch, cursor)

    with pytest.raises(ValueError, match="more than"):
        run_general_query.invoke({"query_text": "SELECT n"})


def test_oversized_result_bytes_is_rejected(monkeypatch):
    rows = [("x" * (MAX_RESULT_BYTES + 1),)]
    cursor = _FakeCursor([("t",)], rows)
    _patch(monkeypatch, cursor)

    with pytest.raises(ValueError, match="too large"):
        run_general_query.invoke({"query_text": "SELECT t"})
