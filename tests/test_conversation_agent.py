from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agents import conversation_agent
from state.state import SupervisorState


class _FakeLLM:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="ok")


def test_convo_includes_actual_user_message(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(conversation_agent, "llm", lambda: fake)

    state = SupervisorState(
        messages=[HumanMessage(content="My name is Rami and I prefer pie charts.")],
        current_task="Acknowledge the user's stated preferences.",
    )

    result = conversation_agent.Convo(state)

    assert result["last_result"].status == "done"
    joined = " ".join(str(getattr(m, "content", "")) for m in fake.messages)
    assert "My name is Rami" in joined
