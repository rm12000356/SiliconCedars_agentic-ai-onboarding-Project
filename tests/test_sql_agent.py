"""
Specialist tests for Sql_agent. The permission-denial path is a pure,
deterministic code check that runs before any model or tool call, so
it's tested here with zero LLM/DB dependency: fast, reliable, no
network flakiness. This is exactly the mechanism verified by the
adversarial test the audit called for: "general user asks for salary,
expect PERMISSION_DENIED, no SQL retry."
"""

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from agents.sql_agent import Sql_agent
from state.state import SupervisorState


def _make_state(task: str) -> SupervisorState:
    return SupervisorState(
        messages=[HumanMessage(content=task)],
        current_task=task,
    )


def _config(permission_level: str) -> RunnableConfig:
    return {"configurable": {"permission_level": permission_level}}


@pytest.mark.parametrize(
    "task",
    [
        "What is the salary of Rami Noueihed?",
        "Show me all employee salaries",
        "Get the credential for user 5",
        "What is the password hash for this account?",
    ],
)
def test_general_permission_blocks_sensitive_requests(task):
    state = _make_state(task)
    result = Sql_agent(state, _config("general"))

    specialist_result = result["last_result"]
    assert specialist_result.status == "failed"
    assert specialist_result.issue == "permission_denied"
    assert specialist_result.source == "sql"


@pytest.mark.integration
def test_general_permission_does_not_block_nonsensitive_keywords():

    state = _make_state("What is the compensation for Rami Noueihed?")
    result = Sql_agent(state, _config("general"))

    specialist_result = result["last_result"]

    assert specialist_result.issue == "permission_denied"


def test_sql_agent_raises_on_missing_current_task():
    state = SupervisorState(messages=[], current_task=None)
    with pytest.raises(RuntimeError, match="current_task=None"):
        Sql_agent(state, _config("general"))