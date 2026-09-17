from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from services.message_utils import (
    CLARIFICATION_ANSWER_FLAG,
    clarification_answers,
    latest_user_request,
)


def _answer(text: str) -> HumanMessage:
    return HumanMessage(
        content=text,
        additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
    )


def test_clarification_answers_collected_in_order():
    messages = [
        HumanMessage(content="show me a chart"),
        AIMessage(content="which kind?"),
        _answer("pie chart"),
        _answer("2024"),
    ]

    assert clarification_answers(messages) == ["pie chart", "2024"]


def test_clarification_answers_ignores_plain_requests():
    assert clarification_answers([HumanMessage(content="hello")]) == []


def test_latest_user_request_still_returns_original():
    messages = [
        HumanMessage(content="show me a chart"),
        AIMessage(content="which kind?"),
        _answer("pie chart"),
    ]

    assert latest_user_request(messages) == "show me a chart"


def test_clarification_answers_scoped_to_current_turn():
    messages = [
        HumanMessage(content="show me a chart"),
        AIMessage(content="which kind?"),
        _answer("pie chart"),
        HumanMessage(content="thanks"),
        HumanMessage(content="now show me sales"),
        AIMessage(content="which kind?"),
        _answer("line chart"),
    ]

    assert clarification_answers(messages) == ["line chart"]
