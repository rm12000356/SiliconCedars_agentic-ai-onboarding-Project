from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agents import visualization_agent
from agents.visualization_agent import Visualization, _resolve_chart_type
from services.message_utils import CLARIFICATION_ANSWER_FLAG
from state.state import SpecialistResult, SupervisorState


def _state(last_result: SpecialistResult | None) -> SupervisorState:
    return SupervisorState(
        messages=[HumanMessage(content="show me a bar chart of sales by region")],
        current_task="Create a chart from the previous result.",
        last_result=last_result,
    )


def test_structured_data_path_sets_chart_path(tmp_path, monkeypatch):
    monkeypatch.setattr(visualization_agent, "OUTPUT_DIR", tmp_path)

    state = _state(
        SpecialistResult(
            source="sql",
            summary="Sales by region",
            status="done",
            structured_data=[
                {"label": "MENA", "value": 1200.5},
                {"label": "EU", "value": 800.0},
            ],
        )
    )

    result = Visualization(state)

    assert result["last_result"].status == "done"
    chart_path = result["chart_path"]
    assert chart_path is not None
    assert Path(chart_path).suffix == ".png"
    assert Path(chart_path).exists()


def test_invalid_structured_data_reports_failure_without_chart_path(tmp_path, monkeypatch):
    monkeypatch.setattr(visualization_agent, "OUTPUT_DIR", tmp_path)

    state = _state(
        SpecialistResult(
            source="sql",
            summary="bad rows",
            status="done",
            structured_data=[{"label": "x", "value": "not-a-number"}],
        )
    )

    result = Visualization(state)

    assert result["last_result"].status == "failed"
    assert result["last_result"].issue == "invalid_chart_spec"
    assert result["chart_path"] is None


def _clarification_state(
    last_result: SpecialistResult | None = None,
) -> SupervisorState:
    return SupervisorState(
        messages=[
            HumanMessage(content="show me a chart of sales by region"),
            AIMessage(content="Which type of chart?"),
            HumanMessage(
                content="pie chart",
                additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
            ),
        ],
        current_task="Create a chart from the previous result.",
        last_result=last_result,
    )


def test_resolve_chart_type_uses_clarification_answer():
    assert _resolve_chart_type(_clarification_state()) == "pie"


def test_resolve_chart_type_answer_overrides_original():
    state = SupervisorState(
        messages=[
            HumanMessage(content="show me a line chart of sales"),
            AIMessage(content="Which type of chart?"),
            HumanMessage(
                content="pie chart",
                additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
            ),
        ],
        current_task="Create a chart from the previous result.",
    )

    assert _resolve_chart_type(state) == "pie"


def test_resolve_chart_type_defaults_to_bar():
    state = SupervisorState(
        messages=[HumanMessage(content="show me a chart")],
        current_task="Create a chart from the previous result.",
    )

    assert _resolve_chart_type(state) == "bar"


def test_resolve_chart_type_ignores_stale_answer():
    state = SupervisorState(
        messages=[
            HumanMessage(content="show me a chart of sales"),
            AIMessage(content="Which type of chart?"),
            HumanMessage(
                content="pie chart",
                additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
            ),
            HumanMessage(content="now show me a line chart of sales"),
        ],
        current_task="Create a chart from the previous result.",
    )

    assert _resolve_chart_type(state) == "line"


def test_resolve_chart_type_accepts_bare_pie_answer():
    state = SupervisorState(
        messages=[
            HumanMessage(content="show me a chart of sales"),
            AIMessage(content="Which type of chart?"),
            HumanMessage(
                content="as a pie",
                additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
            ),
        ],
        current_task="Create a chart from the previous result.",
    )

    assert _resolve_chart_type(state) == "pie"


def test_resolve_chart_type_accepts_bare_line_answer():
    state = SupervisorState(
        messages=[
            HumanMessage(content="show me a chart of sales over time"),
            AIMessage(content="Which type of chart?"),
            HumanMessage(
                content="as a line",
                additional_kwargs={CLARIFICATION_ANSWER_FLAG: True},
            ),
        ],
        current_task="Create a chart from the previous result.",
    )

    assert _resolve_chart_type(state) == "line"


def test_structured_path_uses_clarification_chart_type(tmp_path, monkeypatch):
    monkeypatch.setattr(visualization_agent, "OUTPUT_DIR", tmp_path)
    captured = {}

    def fake_render(spec):
        captured["chart_type"] = spec.chart_type
        path = tmp_path / "clarified.png"
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr(visualization_agent, "_render_chart", fake_render)

    state = _clarification_state(
        SpecialistResult(
            source="sql",
            summary="Sales by region",
            status="done",
            structured_data=[
                {"label": "MENA", "value": 1200.5},
                {"label": "EU", "value": 800.0},
            ],
        )
    )

    result = Visualization(state)

    assert result["last_result"].status == "done"
    assert captured["chart_type"] == "pie"
