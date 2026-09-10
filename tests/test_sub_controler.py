from agents.research.supervisor import Sub_controler, MAX_RESEARCH_ATTEMPTS
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