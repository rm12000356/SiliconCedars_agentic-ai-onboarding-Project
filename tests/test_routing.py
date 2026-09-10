
from types import SimpleNamespace
from typing import cast
import pytest
from graph.routing import route_supervisor, route_sub_supervisor
from state.state import SupervisorState, SubGraphSupervisorState


def test_route_supervisor_returns_valid_route():
    state = SupervisorState(messages=[], next="sql")
    assert route_supervisor(state) == "sql"


def test_route_supervisor_raises_on_none():
    state = SupervisorState(messages=[], next=None)
    with pytest.raises(RuntimeError, match="state.next is None"):
        route_supervisor(state)

def test_route_supervisor_raises_on_corrupted_value():
    fake_state = cast(SupervisorState, SimpleNamespace(next="not_a_real_route"))
    with pytest.raises(RuntimeError, match="State corruption"):
        route_supervisor(fake_state)


def test_route_sub_supervisor_returns_valid_route():
    state = SubGraphSupervisorState(messages=[], task="x", next="researcher")
    assert route_sub_supervisor(state) == "researcher"


def test_route_sub_supervisor_raises_on_none():
    state = SubGraphSupervisorState(messages=[], task="x", next=None)
    with pytest.raises(RuntimeError, match="sub-supervisor state.next is None"):
        route_sub_supervisor(state)


def test_route_sub_supervisor_raises_on_corrupted_value():
    fake_state = cast(
        SubGraphSupervisorState,
        SimpleNamespace(next="not_a_real_subroute"),
    )
    with pytest.raises(RuntimeError, match="State corruption"):
        route_sub_supervisor(fake_state)