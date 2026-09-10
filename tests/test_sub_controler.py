from agents.research.supervisor import Sub_controler, MAX_RESEARCH_ATTEMPTS
from state.state import SubGraphSupervisorState


def test_report_written_forces_end_unconditionally():
    # Even with research_attempts still under the ceiling, report_written
    # must win. This is the exact scenario that caused the infinite loop:
    # forcing "report" after exhausting attempts, then never being able
    # to reach "end" afterward.
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
    # Both conditions true at once: report_written must still win, since
    # it's checked first. This directly documents the fix's ordering
    # requirement, if these ever get reordered, this test catches it.
    state = SubGraphSupervisorState(
        messages=[],
        task="anything",
        research_attempts=MAX_RESEARCH_ATTEMPTS,
        report_written=True,
    )
    result = Sub_controler(state)
    assert result == {"next": "end"}