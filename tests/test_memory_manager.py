from __future__ import annotations

from agents.memory_manager import memory_manager
from state.state import SupervisorState


def test_memory_manager_resets_chart_path_each_turn():
    """
    A chart from a previous turn must not be re-sent: memory_manager is the
    per-turn entry node, so it clears chart_path before the Supervisor runs.
    """
    state = SupervisorState(messages=[], chart_path="outputs/old.png", turn_count=1)

    update = memory_manager(state)

    assert update["chart_path"] is None
