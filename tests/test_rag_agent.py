import pytest
from state.state import SupervisorState
from agents.rag_agent import RAG


@pytest.mark.integration
def test_rag_finds_real_match():
    state = SupervisorState(
        messages=[],
        current_task="What did we learn about SQL security from past projects?",
    )
    result = RAG(state)
    specialist_result = result["last_result"]

    assert specialist_result.status == "done"
    assert specialist_result.source == "rag"
    # Should be grounded in the actual retrieved lesson, not generic
    # SQL-security advice from the model's training knowledge.
    assert "role" in specialist_result.summary.lower() or "postgres" in specialist_result.summary.lower()


@pytest.mark.integration
def test_rag_reports_no_match_honestly():
    state = SupervisorState(
        messages=[],
        current_task="What have we learned about deploying to Antarctica?",
    )
    result = RAG(state)
    specialist_result = result["last_result"]

    assert specialist_result.status == "partial"
    assert specialist_result.issue == "no_matching_documents"


def test_rag_raises_on_missing_current_task():
    state = SupervisorState(messages=[], current_task=None)
    with pytest.raises(RuntimeError, match="current_task=None"):
        RAG(state)