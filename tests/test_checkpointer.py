from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from services.memory import get_checkpointer


def test_memory_backend_returns_checkpointer_and_no_context(monkeypatch):
    """get_checkpointer() returns (checkpointer, saver_context). The memory
    backend needs no context manager, so the second element is None."""
    monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
    memory, saver_context = get_checkpointer()
    assert memory is not None
    assert isinstance(memory, MemorySaver) or hasattr(memory, "get")
    assert saver_context is None


def test_memory_backend_checkpointer_is_usable(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
    memory, _ = get_checkpointer()
    config: RunnableConfig = {"configurable": {"thread_id": "checkpointer-test"}}
    assert memory.get(config) is None