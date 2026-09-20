from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.finalize import Finalize, NO_ANSWER_FALLBACK, OUTAGE_MESSAGE
from state.state import SpecialistResult, SupervisorState


def _config() -> RunnableConfig:
    return RunnableConfig(configurable={})


def test_no_specialist_ran_falls_back_instead_of_echoing_user():
    """
    Supervisor picks "end" as the very first decision of a turn: last_result
    stays None, and the thread's only message is the incoming HumanMessage.
    Finalize must append exactly one fallback AIMessage rather than leaving
    messages[-1] as the user's own message -- chat.py and main.py both print
    messages[-1] as "the assistant's reply".
    """
    state = SupervisorState(
        messages=[HumanMessage(content="hello")],
        current_task=None,
        last_result=None,
        turn_count=1,
    )

    update = Finalize(state, _config())

    assert "messages" in update
    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], AIMessage)
    assert update["messages"][0].content == NO_ANSWER_FALLBACK
    assert update["messages"][0].content != "hello"


def test_specialist_answer_is_appended_exactly_once():
    """
    Locks in the double-append fix: a real, non-empty specialist result must
    produce exactly one AIMessage, not the old positional block plus the new
    content-based block both firing.
    """
    state = SupervisorState(
        messages=[HumanMessage(content="hello")],
        current_task=None,
        last_result=SpecialistResult(
            source="convo", summary="Hi there!", status="done"
        ),
        turn_count=1,
    )

    update = Finalize(state, _config())

    assert "messages" in update
    assert len(update["messages"]) == 1
    assert update["messages"][0].content == "Hi there!"


def test_outage_emits_static_message_without_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("Finalize must not call the LLM during an outage")

    monkeypatch.setattr("agents.finalize.llm", boom)

    state = SupervisorState(
        messages=[HumanMessage(content="hello")],
        current_task="",
        last_result=None,
        turn_count=1,
        outage=True,
    )

    update = Finalize(state, _config())

    assert update["messages"][0].content == OUTAGE_MESSAGE
    assert update["last_result"] is None