from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage

from agents import visualization_agent
from agents.visualization_agent import Visualization
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
