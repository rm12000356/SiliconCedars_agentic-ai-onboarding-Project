from __future__ import annotations

import pytest

from agents.rag_agent import RAG
from state.state import SupervisorState
from langchain_core.messages import HumanMessage

from tests.conftest import requires_llm, requires_db


pytestmark = [pytest.mark.llm, requires_llm, requires_db]


def _state(task: str) -> SupervisorState:
    return SupervisorState(
        messages=[HumanMessage(content=task)],
        current_task=task,
    )


def test_llm_rag_answers_from_internal_lessons():
    """
    Query that should match seeded lessons_learned content
    (SQL security / Postgres role-level access).
    """
    task = "What did we learn about SQL security from past projects?"
    result = RAG(_state(task))
    sr = result["last_result"]

    assert sr.source == "rag"
    assert sr.status == "done", f"expected done, got {sr.status}: {sr.summary} issue={sr.issue}"
    assert sr.issue is None

    summary = sr.summary.lower()
    # Grounded signals from seeded lesson text about role-level restriction
    grounded = [
        "role",
        "postgres",
        "permission",
        "security",
        "sql",
        "grant",
        "validation",
    ]
    assert any(g in summary for g in grounded), (
        f"answer does not look grounded in internal lessons: {sr.summary}"
    )


def test_llm_rag_supervisor_design_lesson():
    """Another in-corpus topic: deterministic guards vs prompt-only loop prevention."""
    task = "What did we learn about preventing agent routing loops?"
    result = RAG(_state(task))
    sr = result["last_result"]

    assert sr.source == "rag"
    # Either a grounded done answer or honest partial is acceptable if retrieval is weak;
    # prefer done with design-related content when chunks match.
    assert sr.status in ("done", "partial"), f"unexpected status: {sr.status}"

    if sr.status == "done":
        summary = sr.summary.lower()
        signals = ["loop", "guard", "deterministic", "prompt", "counter", "history", "supervisor"]
        assert any(s in summary for s in signals), (
            f"done answer missing expected lesson themes: {sr.summary}"
        )


def test_llm_rag_no_relevant_documents():
    """
    Out-of-corpus question must not invent internal policy.
    Expect partial + no_matching_documents when retrieval distances are high.
    """
    task = "What is our company policy for deploying services to Antarctica research stations?"
    result = RAG(_state(task))
    sr = result["last_result"]

    assert sr.source == "rag"
    assert sr.status == "partial", f"expected partial, got {sr.status}: {sr.summary}"
    assert sr.issue == "no_matching_documents"
    assert "no matching" in sr.summary.lower() or "not found" in sr.summary.lower()


def test_llm_rag_does_not_claim_external_facts_as_internal():
    """
    When nothing is retrieved, the agent must not return status=done
    with fabricated internal knowledge.
    """
    task = "According to our internal handbook, what is the exact coffee budget per employee in euros?"
    result = RAG(_state(task))
    sr = result["last_result"]

    assert sr.source == "rag"
    # Strong expectation: no match path
    if sr.status == "done":
        # If retrieval somehow matched something unrelated, still must not invent a euro budget
        summary = sr.summary.lower()
        assert not re_search_money_claim(summary), (
            f"RAG invented a specific budget figure: {sr.summary}"
        )
    else:
        assert sr.status == "partial"
        assert sr.issue == "no_matching_documents"


def re_search_money_claim(text: str) -> bool:
    import re
    # Heuristic: a concrete euro amount claim
    return bool(re.search(r"€\s*\d+|\d+\s*euros?", text))