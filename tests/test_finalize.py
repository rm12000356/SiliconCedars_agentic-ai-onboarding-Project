from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.finalize import (
    CHART_SYNTHESIS_NOTE,
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


def test_not_chartable_step_uses_distinct_message():
    state = SupervisorState(
        messages=[HumanMessage(content="chart sales")],
        plan=[
            PlanItem(
                route="visu",
                task="chart",
                status="skipped",
                issue="not_chartable",
                result_summary="The chart was skipped because the result had no chartable data.",
            )
        ],
        turn_count=1,
    )

    update = Finalize(state, _config())
    content = update["messages"][0].content

    assert "chart" in content.lower()
    assert "label and a number" in content.lower()


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


# --- chart handling ---------------------------------------------------------

_SALES_ROWS = [{"label": "EU", "value": 100}, {"label": "MENA", "value": 200}]


def _capture_model(captured, content="Synthesized answer."):
    class _Response:
        pass

    response = _Response()
    response.content = content

    class _Model:
        def invoke(self, messages):
            captured["messages"] = messages
            return response

    return _Model()


def test_single_answer_plus_chart_skips_llm(monkeypatch):
    """One non-chart answer plus a chart: no synthesis, chart line appended."""

    def boom(*_a, **_k):
        raise AssertionError("one non-chart answer needs no synthesis")

    monkeypatch.setattr("agents.finalize.llm", boom)

    state = SupervisorState(
        messages=[HumanMessage(content="count employees and chart sales")],
        plan=[
            PlanItem(
                route="sql",
                task="count employees",
                status="done",
                result_summary="The employees table contains 2 employees.",
            ),
            PlanItem(
                route="sql",
                task="sales by region",
                status="done",
                result_summary="Region totals: EU 100, MENA 200.",
                structured_data=_SALES_ROWS,
            ),
            PlanItem(
                route="visu",
                task="chart sales",
                status="done",
                data_source="database",
                result_summary="Here's the chart: Sales by region.",
            ),
        ],
        chart_path="outputs/sales.png",
        turn_count=1,
    )

    update = Finalize(state, _config())
    content = update["messages"][0].content

    assert content == (
        "The employees table contains 2 employees.\n\n"
        "Here's the chart: Sales by region."
    )
    assert "EU 100" not in content


def test_two_answers_plus_chart_synthesizes_and_appends_chart(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "agents.finalize.llm",
        lambda *a, **k: _capture_model(
            captured, "There are 2 employees. The policy is documented."
        ),
    )

    state = SupervisorState(
        messages=[HumanMessage(content="count, policy, and chart sales")],
        plan=[
            PlanItem(
                route="sql",
                task="count employees",
                status="done",
                result_summary="The employees table contains 2 employees.",
            ),
            PlanItem(
                route="rag",
                task="remote work policy",
                status="done",
                result_summary="The policy is documented.",
            ),
            PlanItem(
                route="sql",
                task="sales by region",
                status="done",
                result_summary="Region totals: EU 100, MENA 200.",
                structured_data=_SALES_ROWS,
            ),
            PlanItem(
                route="visu",
                task="chart sales",
                status="done",
                data_source="database",
                result_summary="Here's the chart: Sales by region.",
            ),
        ],
        chart_path="outputs/sales.png",
        turn_count=1,
    )

    update = Finalize(state, _config())
    content = update["messages"][0].content

    assert content.startswith("There are 2 employees. The policy is documented.")
    assert content.endswith("Here's the chart: Sales by region.")

    system, human = captured["messages"]
    assert CHART_SYNTHESIS_NOTE in system.content
    assert "2 employees" in human.content
    assert "policy" in human.content.lower()
    assert "Region totals" not in human.content
    assert "EU" not in human.content


def test_inline_visu_does_not_exclude_preceding_chartable_sql(monkeypatch):
    """An inline chart uses typed-in numbers; a preceding chartable sql step
    is a separate answer and must not be dropped."""
    captured: dict = {}
    monkeypatch.setattr(
        "agents.finalize.llm",
        lambda *a, **k: _capture_model(
            captured, "Engineering 5, Sales 3. Total sales are 1000."
        ),
    )

    state = SupervisorState(
        messages=[HumanMessage(content="employees by dept and chart 10 and 20")],
        plan=[
            PlanItem(
                route="sql",
                task="employees by department",
                status="done",
                result_summary="Engineering 5, Sales 3.",
                structured_data=[
                    {"label": "Engineering", "value": 5},
                    {"label": "Sales", "value": 3},
                ],
            ),
            PlanItem(
                route="sql",
                task="total sales",
                status="done",
                result_summary="Total sales are 1000.",
            ),
            PlanItem(
                route="visu",
                task="chart these numbers: 10 and 20",
                status="done",
                data_source="inline",
                result_summary="Here's the chart: Inline numbers.",
            ),
        ],
        chart_path="outputs/inline.png",
        turn_count=1,
    )

    update = Finalize(state, _config())
    human = captured["messages"][1].content

    assert "Engineering 5, Sales 3." in human
    assert "Total sales are 1000." in human
    assert update["messages"][0].content.endswith(
        "Here's the chart: Inline numbers."
    )


def test_chart_only_uses_visu_summary_without_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("chart-only turn must not synthesize")

    monkeypatch.setattr("agents.finalize.llm", boom)

    state = SupervisorState(
        messages=[HumanMessage(content="chart sales")],
        plan=[
            PlanItem(
                route="sql",
                task="sales",
                status="done",
                result_summary="Region totals: EU 100, MENA 200.",
                structured_data=_SALES_ROWS,
            ),
            PlanItem(
                route="visu",
                task="chart",
                status="done",
                data_source="database",
                result_summary="Here's the chart: Sales.",
            ),
        ],
        chart_path="outputs/sales.png",
        turn_count=1,
    )

    update = Finalize(state, _config())

    assert update["messages"][0].content == "Here's the chart: Sales."


def test_inline_chart_only_uses_summary_without_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("chart-only turn must not synthesize")

    monkeypatch.setattr("agents.finalize.llm", boom)

    state = SupervisorState(
        messages=[HumanMessage(content="pie: 10 and 20")],
        plan=[
            PlanItem(
                route="visu",
                task="pie: 10 and 20",
                status="done",
                data_source="inline",
                result_summary="Here's the chart: Pie.",
            )
        ],
        chart_path="outputs/pie.png",
        turn_count=1,
    )

    update = Finalize(state, _config())

    assert update["messages"][0].content == "Here's the chart: Pie."


def test_chart_note_absent_without_chart(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "agents.finalize.llm",
        lambda *a, **k: _capture_model(captured, "combined"),
    )

    Finalize(_two_step_state(), _config())

    assert CHART_SYNTHESIS_NOTE not in captured["messages"][0].content


def test_sql_plus_failed_visu_still_synthesizes(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "agents.finalize.llm",
        lambda *a, **k: _capture_model(captured, "The chart could not be built."),
    )

    state = SupervisorState(
        messages=[HumanMessage(content="count employees and chart sales")],
        plan=[
            PlanItem(
                route="sql",
                task="count employees",
                status="done",
                result_summary="The employees table contains 2 employees.",
            ),
            PlanItem(
                route="visu",
                task="chart sales",
                status="skipped",
                issue="no_data_for_chart",
                result_summary="The chart was skipped because the data was unavailable.",
            ),
        ],
        turn_count=1,
    )

    update = Finalize(state, _config())

    assert "messages" in captured
    assert "2 employees" in captured["messages"][1].content
    assert update["messages"][0].content == "The chart could not be built."