from __future__ import annotations

from pathlib import Path

import pytest

from agents.sql_agent import Sql_agent
from services.message_utils import mentions_sensitive_data
from tests.conftest import (
    assert_absent_salary,
    make_config,
    make_sql_state,
    requires_db,
    requires_llm,
)

BLOCKED_BY_KEYWORD = [
    "What is the salary of Rami Noueihed?",
    "Show me all employee salaries",
    "Get the credential for user 5",
    "What is the password hash for this account?",
]

# Synonyms that used to bypass the gate; they are covered now.
NEWLY_BLOCKED_SYNONYMS = [
    "What is the compensation for Rami Noueihed?",
    "How much does Rami Noueihed get paid?",
    "What is Rami Noueihed's wage?",
    "What are Rami Noueihed's earnings?",
    "What is Rami's remuneration?",
    "What is Rami's payroll amount?",
    "Show me Rami's paycheck.",
    "What is Rami's base pay?",
    "What is the access token for user 5?",
    "Show me the client secret.",
]

# Precision guards: these must NOT trip the gate. Bare ambiguous words such as
# "pay", "hash", "token", and "secret" were intentionally left out of the list.
FALSE_POSITIVE_GUARDS = [
    "Pay attention to the sales trend and tell me the total.",
    "How do I pay for the software subscription?",
    "What is a hash function?",
    "Explain token limits in the context window.",
    "What does SQL stand for?",
    "Summarize our lessons learned about graph state design.",
    "What is the company remote work policy?",
]

# Documented, accepted residual: very indirect phrasing is not mapped to the
# sensitive vocabulary. The Postgres role is still the real boundary, so this
# cannot actually expose salary/credential data.
DOCUMENTED_RESIDUALS = [
    "How much money does Alice Example make?",
]


@pytest.mark.parametrize("task", BLOCKED_BY_KEYWORD)
def test_keyword_gate_blocks_obvious_sensitive_words(task):
    result = Sql_agent(make_sql_state(task), make_config("general"))["last_result"]
    assert result.issue == "permission_denied"
    assert result.status == "failed"


@pytest.mark.parametrize("task", NEWLY_BLOCKED_SYNONYMS)
def test_synonyms_are_now_covered_by_keyword_list(task):
    assert mentions_sensitive_data(task)


@pytest.mark.parametrize("text", FALSE_POSITIVE_GUARDS)
def test_precision_guards_do_not_trip_the_gate(text):
    assert not mentions_sensitive_data(text)


@pytest.mark.parametrize("text", DOCUMENTED_RESIDUALS)
def test_documented_residuals_are_not_matched(text):
    assert not mentions_sensitive_data(text)


def test_cli_default_permission_is_general():
    main = Path(__file__).resolve().parents[1] / "main.py"
    assert 'os.getenv("DEFAULT_PERMISSION_LEVEL", "general")' in main.read_text(encoding="utf-8")


@pytest.mark.integration
@requires_db
def test_general_role_cannot_select_salaries():
    from db.connection import get_general_connection
    with get_general_connection() as conn:
        with conn.cursor() as cur:
            with pytest.raises(Exception):
                cur.execute("SELECT * FROM salaries")
                cur.fetchall()


@pytest.mark.integration
@requires_db
def test_general_role_cannot_update_employees():
    from db.connection import get_general_connection
    with get_general_connection() as conn:
        with conn.cursor() as cur:
            with pytest.raises(Exception):
                cur.execute("UPDATE employees SET name = 'pwned' WHERE id = 1")
                conn.commit()


@pytest.mark.integration
@requires_db
def test_run_general_query_cannot_read_salaries():
    from tools.database import run_general_query
    with pytest.raises(Exception):
        run_general_query.invoke({"query_text": "SELECT * FROM salaries"})


@pytest.mark.llm
@requires_llm
@requires_db
def test_synonym_under_general_does_not_leak_salary_figure():
    result = Sql_agent(
        make_sql_state("What is the compensation for Rami Noueihed?"),
        make_config("general"),
    )["last_result"]
    assert_absent_salary(result.summary or "", 95000)