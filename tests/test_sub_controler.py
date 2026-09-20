from langchain_core.messages import AIMessage

from agents.research.supervisor import Sub_controler, MAX_RESEARCH_ATTEMPTS
from agents.research.report_writer import Report_W
from state.state import SubGraphSupervisorState


def test_report_written_forces_end_unconditionally():

    state = SubGraphSupervisorState(
        messages=[],
        task="anything",
        research_attempts=1,
        report_written=True,
    )
    result = Sub_controler(state)
    assert result == {"next": "end"}


def test_max_attempts_forces_report_when_not_yet_written():
    state = SubGraphSupervisorState(
        messages=[],
        task="anything",
        research_attempts=MAX_RESEARCH_ATTEMPTS,
        report_written=False,
    )
    result = Sub_controler(state)
    assert result == {"next": "report"}


def test_report_written_takes_priority_over_max_attempts():
    
    state = SubGraphSupervisorState(
        messages=[],
        task="anything",
        research_attempts=MAX_RESEARCH_ATTEMPTS,
        report_written=True,
    )
    result = Sub_controler(state)
    assert result == {"next": "end"}


def test_no_research_yet_routes_researcher_without_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("no LLM call is needed before any research")

    monkeypatch.setattr("agents.research.supervisor.llm", boom)
    state = SubGraphSupervisorState(messages=[], task="x")

    result = Sub_controler(state)

    assert result == {"next": "researcher", "research_attempts": 1}


class _FailingModel:
    def with_structured_output(self, _schema, **_kwargs):
        return self

    def invoke(self, _messages):
        raise RuntimeError("provider down")


def test_controller_provider_error_forces_report(monkeypatch):
    monkeypatch.setattr(
        "agents.research.supervisor.llm", lambda *a, **k: _FailingModel()
    )
    state = SubGraphSupervisorState(
        messages=[],
        task="x",
        research_attempts=1,
        research_messages=[AIMessage(content="short note")],
    )

    result = Sub_controler(state)

    assert result == {"next": "report"}


def test_report_writer_provider_error_degrades(monkeypatch):
    monkeypatch.setattr(
        "agents.research.report_writer.llm", lambda *a, **k: _FailingModel()
    )
    state = SubGraphSupervisorState(
        messages=[],
        task="x",
        research_messages=[AIMessage(content="some material")],
    )

    result = Report_W(state)

    assert result["research_succeeded"] is False
    assert result["report_written"] is True