from langgraph.checkpoint.memory import MemorySaver
from services.memory import get_checkpointer


def test_memory_backend_works_as_context_manager(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
    cm = get_checkpointer()
    with cm as memory:
        assert memory is not None
        assert isinstance(memory, MemorySaver) or hasattr(memory, "get")


def test_memory_backend_context_manager_exits_cleanly(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
    with get_checkpointer() as memory:
        assert memory is not None