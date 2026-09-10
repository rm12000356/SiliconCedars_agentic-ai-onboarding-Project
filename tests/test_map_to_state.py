from agents.supervisor import map_to_state
from state.structure_output import SupervisorDecision

def test_map_to_state_clears_last_result_for_ordinary_routes():
    decision = SupervisorDecision(next="sql", current_task="do something")
    update = map_to_state(decision)
    assert update["last_result"] is None
    assert update["next"] == "sql"
    assert update["current_task"] == "do something"

def test_map_to_state_preserves_last_result_on_end():
    decision = SupervisorDecision(next="end", current_task="")
    update = map_to_state(decision)
    assert "last_result" not in update

def test_map_to_state_preserves_last_result_on_visu():
    decision = SupervisorDecision(next="visu", current_task="chart it")
    update = map_to_state(decision)
    assert "last_result" not in update