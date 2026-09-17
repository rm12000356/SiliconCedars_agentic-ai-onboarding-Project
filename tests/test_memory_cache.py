from __future__ import annotations

import services.memory as memory


class _FakeCursor:
    def __init__(self, executed: list) -> None:
        self._executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, *args, **kwargs):
        self._executed.append(args)


class _FakeConnection:
    def __init__(self, executed: list) -> None:
        self._executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return _FakeCursor(self._executed)

    def commit(self):
        pass


class _FakeConnectionManager:
    def __init__(self, executed: list) -> None:
        self._executed = executed

    def __enter__(self):
        return _FakeConnection(self._executed)

    def __exit__(self, *args):
        return False


def test_ensure_memory_table_runs_ddl_once(monkeypatch):
    executed: list = []
    monkeypatch.setattr(memory, "_memory_table_ready", False)
    monkeypatch.setattr(
        memory,
        "get_elevated_connection",
        lambda: _FakeConnectionManager(executed),
    )

    memory._ensure_memory_table()
    memory._ensure_memory_table()
    memory._ensure_memory_table()

    assert len(executed) == 1
