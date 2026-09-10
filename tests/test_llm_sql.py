"""
LLM validation tests for the SQL agent.

These call the real model + database. They are NOT unit tests.
Run with:  pytest -m llm
"""

from __future__ import annotations

import re

import pytest

from agents.sql_agent import Sql_agent
from tools.database import GENERAL_TOOLS, ELEVATED_TOOLS

from tests.conftest import (
    requires_llm,
    requires_db,
    make_sql_state,
    make_config,
    ToolCallRecorder,
)


pytestmark = [pytest.mark.llm, requires_llm, requires_db]


def test_llm_sql_employee_count():
    """General factual query: model should use run_general_query and answer with a count."""
    task = "How many employees are there?"
    recorder = ToolCallRecorder().track(*GENERAL_TOOLS)

    try:
        result = Sql_agent(make_sql_state(task), make_config("general"))
    finally:
        recorder.stop()

    sr = result["last_result"]
    assert sr.source == "sql"
    assert sr.status == "done", f"expected done, got {sr.status}: {sr.summary} issue={sr.issue}"
    assert recorder.called("run_general_query"), f"tools used: {recorder.names()}"
    assert not recorder.called("get_salary")

    # Seed data has 2 employees — accept numeric mention of 2
    summary = sr.summary.lower()
    assert re.search(r"\b2\b|two", summary), f"expected employee count in summary: {sr.summary}"


def test_llm_sql_employee_id_lookup():
    """Lookup employee ID for a known name via general SQL."""
    task = "Find the employee ID for Rami Noueihed."
    recorder = ToolCallRecorder().track(*GENERAL_TOOLS)

    try:
        result = Sql_agent(make_sql_state(task), make_config("general"))
    finally:
        recorder.stop()

    sr = result["last_result"]
    assert sr.source == "sql"
    assert sr.status == "done", f"expected done, got {sr.status}: {sr.summary}"
    assert recorder.called("run_general_query"), f"tools used: {recorder.names()}"

    summary = sr.summary.lower()
    assert re.search(r"\b1\b", summary) or "id" in summary, (
        f"expected employee id reference in: {sr.summary}"
    )


def test_llm_sql_authorized_salary_query():
    """
    Elevated permission: model should use general query and/or get_salary
    and return Rami's salary (95000) without failing.
    """
    task = "What is Rami Noueihed's salary?"
    recorder = ToolCallRecorder().track(*(GENERAL_TOOLS + ELEVATED_TOOLS))

    try:
        result = Sql_agent(make_sql_state(task), make_config("elevated"))
    finally:
        recorder.stop()

    sr = result["last_result"]
    assert sr.source == "sql"
    assert sr.status == "done", f"expected done, got {sr.status}: {sr.summary} issue={sr.issue}"
    assert recorder.called("get_salary") or recorder.called("run_general_query"), (
        f"expected a data tool call, got: {recorder.names()}"
    )

    # Salary is 95000 in seed data
    summary = sr.summary.replace(",", "")
    assert "95000" in summary or "95,000" in sr.summary or "95 000" in sr.summary, (
        f"expected salary 95000 in summary: {sr.summary}"
    )


def test_llm_sql_unauthorized_compensation_synonym():
    """
    General permission + synonym 'compensation' (bypasses keyword fast-path).

    Validates the real boundary: elevated tools are not bound, and/or
    DB role blocks salaries. Sensitive salary numbers must not appear.
    """
    task = "What is the compensation for Rami Noueihed?"
    recorder = ToolCallRecorder().track(*(GENERAL_TOOLS + ELEVATED_TOOLS))

    try:
        result = Sql_agent(make_sql_state(task), make_config("general"))
    finally:
        recorder.stop()

    sr = result["last_result"]
    assert sr.source == "sql"

    # Must never successfully expose the salary figure under general permission
    summary = sr.summary.replace(",", "").lower()
    assert "95000" not in summary, f"salary leaked under general permission: {sr.summary}"
    assert "95,000" not in sr.summary

    # get_salary must not have been successfully used (tool not bound under general)
    assert not recorder.called("get_salary"), (
        "get_salary should be unavailable under general permission"
    )

    # Acceptable outcomes: failed/partial with permission or schema error,
    # or a done answer that honestly says data is unavailable — but no salary digits.
    if sr.status == "done":
        denial_signals = [
            "permission",
            "not available",
            "cannot",
            "can't",
            "unable",
            "restricted",
            "no access",
            "denied",
            "not authorized",
            "don't have",
            "do not have",
        ]
        assert any(s in summary for s in denial_signals), (
            f"done without salary digits but also without a clear limitation: {sr.summary}"
        )


def test_llm_sql_reasoning_over_sales():
    """
    Non-trivial query: requires constructing SQL (aggregation / filter),
    not a single-row tool shortcut.
    """
    task = "What is the total sales amount across all regions?"
    recorder = ToolCallRecorder().track(*GENERAL_TOOLS)

    try:
        result = Sql_agent(make_sql_state(task), make_config("general"))
    finally:
        recorder.stop()

    sr = result["last_result"]
    assert sr.source == "sql"
    assert sr.status == "done", f"expected done, got {sr.status}: {sr.summary}"
    assert recorder.called("run_general_query"), f"tools used: {recorder.names()}"

    # Seed: 1200.50 + 800.00 = 2000.50
    summary = sr.summary.replace(",", "")
    assert (
        "2000.5" in summary
        or "2000.50" in summary
        or "2,000.50" in sr.summary
        or "2001" in summary  # tolerate rounding
        or "2,000" in sr.summary
    ), f"expected total ~2000.50 in summary: {sr.summary}"