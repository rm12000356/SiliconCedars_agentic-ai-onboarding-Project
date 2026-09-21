from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.finalize import (
    Finalize,
    NO_ANSWER_FALLBACK,
    OUTAGE_MESSAGE,
    TURN_CUT_SHORT_NOTE,
    _might_contain_memorable_info,
)
from state.state import PlanItem, SpecialistResult, SupervisorState


def _config() -> RunnableConfig:
    return RunnableConfig(configurable={})


def test_no_specialist_ran_falls_back_instead_of_echoing_user():
    """
    Supervisor picks "end" as the very first decision of a turn: last_result
    stays None, and the thread's only message is the incoming HumanMessage.
    Finalize must append exactly one fallback AIMessage rather than leaving
    messages[-1] as the user's own message -- chat.py and main.py both print
    messages[-1] as "the assistant's reply".
    """
    state = SupervisorState(
        messages=[HumanMessage(content="hello")],
        current_task=None,
        last_result=None,
        turn_count=1,
    )

    update = Finalize(state, _config())

    assert "messages" in update
    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], AIMessage)
    assert update["messages"][0].content == NO_ANSWER_FALLBACK
    assert update["messages"][0].content != "hello"


def test_specialist_answer_is_appended_exactly_once():
    """
    Locks in the double-append fix: a real, non-empty specialist result must
    produce exactly one AIMessage, not the old positional block plus the new
    content-based block both firing.
    """
    state = SupervisorState(
        messages=[HumanMessage(content="hello")],
        current_task=None,
        last_result=SpecialistResult(
            source="convo", summary="Hi there!", status="done"
        ),
        turn_count=1,
    )

    update = Finalize(state, _config())

    assert "messages" in update
    assert len(update["messages"]) == 1
    assert update["messages"][0].content == "Hi there!"


def _two_step_state() -> SupervisorState:
    return SupervisorState(
        messages=[HumanMessage(content="count employees and the policy")],
        plan=[
            PlanItem(
                route="sql", task="count", status="done", result_summary="2 employees"
            ),
            PlanItem(
                route="rag",
                task="policy",
                status="done",
                result_summary="The policy is documented.",
            ),
        ],
        turn_count=1,
    )


def test_multiple_plan_results_are_synthesized(monkeypatch):
    class _Response:
        content = "There are 2 employees, and the policy is documented."

    class _Model:
        def invoke(self, _messages):
            return _Response()

    monkeypatch.setattr("agents.finalize.llm", lambda *a, **k: _Model())

    update = Finalize(_two_step_state(), _config())

    assert (
        update["messages"][0].content
        == "There are 2 employees, and the policy is documented."
    )


def test_synthesis_failure_falls_back_to_concatenation(monkeypatch):
    class _Model:
        def invoke(self, _messages):
            raise RuntimeError("provider down")

    monkeypatch.setattr("agents.finalize.llm", lambda *a, **k: _Model())

    update = Finalize(_two_step_state(), _config())
    content = update["messages"][0].content

    assert "2 employees" in content
    assert "The policy is documented." in content


def test_outage_emits_static_message_without_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("Finalize must not call the LLM during an outage")

    monkeypatch.setattr("agents.finalize.llm", boom)

    state = SupervisorState(
        messages=[HumanMessage(content="hello")],
        current_task="",
        last_result=None,
        turn_count=1,
        outage=True,
    )

    update = Finalize(state, _config())

    assert update["messages"][0].content == OUTAGE_MESSAGE
    assert update["last_result"] is None


def test_failed_result_never_leaks_raw_issue_text():
    state = SupervisorState(
        messages=[HumanMessage(content="show salaries")],
        current_task="salaries",
        last_result=SpecialistResult(
            source="sql",
            summary="The requested data could not be found in the database.",
            status="failed",
            issue='table_or_schema_missing: relation "salaries" does not exist',
        ),
        turn_count=1,
    )

    update = Finalize(state, _config())
    content = update["messages"][0].content

    assert "does not exist" not in content
    assert "table_or_schema_missing" not in content


def test_known_issue_uses_user_facing_message():
    state = SupervisorState(
        messages=[HumanMessage(content="show salaries")],
        current_task="salaries",
        last_result=SpecialistResult(
            source="sql",
            summary="Salary and credential data require elevated permissions.",
            status="failed",
            issue="permission_denied",
        ),
        turn_count=1,
    )

    update = Finalize(state, _config())

    assert "permission" in update["messages"][0].content.lower()


def test_skipped_step_uses_user_facing_message():
    state = SupervisorState(
        messages=[HumanMessage(content="chart sales")],
        plan=[
            PlanItem(
                route="visu",
                task="chart",
                status="skipped",
                issue="no_data_for_chart",
                result_summary="The chart was skipped because the data was unavailable.",
            )
        ],
        turn_count=1,
    )

    update = Finalize(state, _config())
    content = update["messages"][0].content

    assert "chart" in content.lower()
    assert "data it needed" in content.lower()


def test_plan_note_is_appended():
    state = SupervisorState(
        messages=[HumanMessage(content="do many things")],
        plan=[
            PlanItem(
                route="sql", task="count", status="done", result_summary="2 employees"
            )
        ],
        turn_count=1,
        plan_note="I focused on the first 3 of 5 requested steps. Not covered: x.",
    )

    update = Finalize(state, _config())
    content = update["messages"][0].content

    assert "2 employees" in content
    assert "Not covered" in content


def test_turn_cut_short_note_is_appended_once():
    state = SupervisorState(
        messages=[HumanMessage(content="do many things")],
        plan=[
            PlanItem(
                route="sql", task="count", status="done", result_summary="2 employees"
            )
        ],
        turn_count=1,
        turn_cut_short=True,
    )

    update = Finalize(state, _config())
    content = update["messages"][0].content

    assert "2 employees" in content
    assert content.count(TURN_CUT_SHORT_NOTE) == 1


def test_bare_im_does_not_trigger_memory_extraction():
    assert not _might_contain_memorable_info("I'm happy with the sales numbers.")


def test_explicit_memory_intent_triggers_extraction():
    assert _might_contain_memorable_info("Remember that I prefer bar charts.")