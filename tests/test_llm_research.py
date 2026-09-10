"""
LLM validation tests for the research subgraph.

Uses real web_search / fetch_page tools and the configured model.
Run with:  pytest -m llm
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from agents.research.researcher import Research, MAX_ITERATIONS, MAX_SEARCH_ATTEMPTS
from agents.research.report_writer import Report_W
from agents.research.supervisor import Sub_controler, MAX_RESEARCH_ATTEMPTS
from agents.research.research_node import make_research_node
from graph.workflow import sub_workflow
from state.state import SupervisorState, SubGraphSupervisorState
from tools.web_search import web_search, fetch_page

from tests.conftest import requires_llm, ToolCallRecorder


pytestmark = [pytest.mark.llm, requires_llm]


def test_llm_research_uses_web_search_and_terminates():
    """
    Full research node: external question should search the web,
    produce a result, and finish without hanging.
    """
    subgraph = sub_workflow()
    node = make_research_node(subgraph)

    state = SupervisorState(
        messages=[HumanMessage(content="What is the capital of France?")],
        current_task="What is the capital of France?",
    )

    recorder = ToolCallRecorder().track(web_search, fetch_page)
    try:
        result = node(state)
    finally:
        recorder.stop()

    sr = result["last_result"]
    assert sr.source == "research"
    assert sr.status in ("done", "failed"), f"unexpected status: {sr.status}"

    # Should have attempted a search for an external factual question
    assert recorder.called("web_search") or sr.status == "failed", (
        f"expected web_search for external query; tools={recorder.names()} status={sr.status}"
    )

    if sr.status == "done":
        summary = sr.summary.lower()
        assert "paris" in summary, f"expected Paris in research answer: {sr.summary}"


def test_llm_researcher_single_pass_bounded():
    """
    Research() alone: respects iteration / search attempt limits
    and returns a message rather than looping forever.
    """
    state = SubGraphSupervisorState(
        messages=[],
        task="Briefly: what is LangGraph used for?",
        research_attempts=0,
    )

    recorder = ToolCallRecorder().track(web_search, fetch_page)
    try:
        result = Research(state)
    finally:
        recorder.stop()

    assert "messages" in result
    assert len(result["messages"]) >= 1

    search_calls = [c for c in recorder.names() if c == "web_search"]
    assert len(search_calls) <= MAX_SEARCH_ATTEMPTS + 1, (
        f"too many web_search calls: {len(search_calls)} (limit {MAX_SEARCH_ATTEMPTS})"
    )


def test_llm_report_writer_sets_report_written():
    """Report writer must always set report_written so the controller can end."""
    state = SubGraphSupervisorState(
        messages=[
            HumanMessage(
                content="Key findings: Paris is the capital of France. Sources: Example Encyclopedia (https://example.com)"
            )
        ],
        task="What is the capital of France?",
    )
    result = Report_W(state)

    assert result.get("report_written") is True
    assert "messages" in result and len(result["messages"]) >= 1
    assert "research_succeeded" in result


def test_llm_sub_controler_ends_after_report_written_without_llm():
    """
    Structural guard (no LLM needed when report_written is set).
    Kept here as a smoke check that the research pipeline's stop condition
    remains intact under the llm marker suite.
    """
    state = SubGraphSupervisorState(
        messages=[],
        task="anything",
        research_attempts=1,
        report_written=True,
    )
    result = Sub_controler(state)
    assert result == {"next": "end"}


def test_llm_full_research_pipeline_stops_within_attempt_budget():
    """
    Invoke the compiled research subgraph and ensure it returns
    (does not infinite-loop past MAX_RESEARCH_ATTEMPTS).
    """
    subgraph = sub_workflow()

    sub_input = SubGraphSupervisorState(
        messages=[],
        task="What year was the Python programming language first released?",
    )

    # Hard safety: LangGraph recursion limit acts as backstop
    result = subgraph.invoke(
        sub_input,
        config={"recursion_limit": 20},
    )

    assert result.get("report_written") is True or result.get("next") in (None, "end") or (
        result.get("research_attempts", 0) <= MAX_RESEARCH_ATTEMPTS + 1
    ), f"research subgraph did not terminate cleanly: keys={list(result.keys())}"

    assert "messages" in result
    assert len(result["messages"]) >= 1