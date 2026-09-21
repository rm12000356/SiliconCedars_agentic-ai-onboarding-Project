"""
LLM routing-precision tests for the supervisor residual path.

First-turn requests (no last_result) fall through to the real model.
Each route has ~20 prompts.

Run:  pytest llm tests/test_llm_supervisor_routing.py -m llm -v -s
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

import pytest

from agents.supervisor import (
    MAX_PLAN_STEPS,
    _validate_plan,
    gather_context,
    get_supervisor_decision,
    get_workflow_plan,
)
from services.llm import llm
from state.state import SupervisorState

from tests.conftest import requires_llm


pytestmark = [pytest.mark.llm, requires_llm]

MIN_ACCURACY = {
    "sql": 0.90,
    "rag": 0.90,
    "research": 0.90,
    "visu": 0.90,
    "convo": 0.90,
    "clarification": 0.90,
}

SQL_CASES = [
    ("How many employees are there?", frozenset({"sql"})),
    ("How many sales records are in the database?", frozenset({"sql"})),
    ("Show me the total salary of every employee.", frozenset({"sql"})),
    ("What is Alice Example's department?", frozenset({"sql"})),
    ("List all employees in Engineering.", frozenset({"sql"})),
    ("What is the total sales amount?", frozenset({"sql"})),
    ("Show sales totals grouped by region.", frozenset({"sql"})),
    ("Find the employee ID for Rami Noueihed.", frozenset({"sql"})),
    ("How many sales happened in the MENA region?", frozenset({"sql"})),
    ("What is Rami Noueihed's salary?", frozenset({"sql"})),
    ("Count employees per department.", frozenset({"sql"})),
    ("What is the latest sale_date in the sales table?", frozenset({"sql"})),
    ("Sum of sales in the EU region.", frozenset({"sql"})),
    ("Which employees are in the Sales department?", frozenset({"sql"})),
    ("Return every row from the employees table.", frozenset({"sql"})),
    ("What is the average sale amount?", frozenset({"sql"})),
    ("How many rows are in lessons_learned?", frozenset({"sql"})),
    ("Show employee names and departments.", frozenset({"sql"})),
    ("Get the credential record for user id 1.", frozenset({"sql"})),
    ("List sales with amount greater than 1000.", frozenset({"sql"})),
]

RAG_CASES = [
    ("What is the company remote work policy?", frozenset({"rag"})),
    ("Explain the internal onboarding process for new hires.", frozenset({"rag"})),
    ("What did we learn about SQL security from past internal projects?", frozenset({"rag"})),
    ("What did we learn internally about preventing agent routing loops?", frozenset({"rag"})),
    ("Summarize our internal lessons on LangGraph state design.", frozenset({"rag"})),
    ("What is our internal policy on database access?", frozenset({"rag"})),
    ("How should we split SQL permissions according to internal docs?", frozenset({"rag"})),
    ("What did the Internal Onboarding Bot v1 project teach us?", frozenset({"rag"})),
    ("Company procedure for handling sensitive credentials.", frozenset({"rag"})),
    ("What are our internal architecture guidelines for multi-agent systems?", frozenset({"rag"})),
    ("Explain our internal lessons about RAG fallback behavior.", frozenset({"rag"})),
    ("What does internal documentation say about turn counters and memory?", frozenset({"rag"})),
    ("Internal process notes from the Customer Support Assistant project.", frozenset({"rag"})),
    ("What organizational procedures exist for new employee onboarding?", frozenset({"rag"})),
    ("Company policy on who can query salary data, from internal docs.", frozenset({"rag"})),
    ("What did we learn from the Customer Support Assistant internally?", frozenset({"rag"})),
    ("According to internal guidance, should the model decide authorization?", frozenset({"rag"})),
    ("What tags do we use on internal lessons_learned documents?", frozenset({"rag"})),
    ("Describe the internal lesson about prompt-only loop prevention.", frozenset({"rag"})),
    ("What is the official internal guidance on supervisor design?", frozenset({"rag"})),
]

RESEARCH_CASES = [
    ("Research the latest market share of Tesla in 2025.", frozenset({"research"})),
    ("Find recent news about OpenAI's latest model releases.", frozenset({"research"})),
    ("Look up the capital of France on the public web.", frozenset({"research"})),
    ("Search the web for the current price of gold.", frozenset({"research"})),
    ("Research who invented the Python programming language.", frozenset({"research"})),
    ("Look up recent public news about NVIDIA GPUs.", frozenset({"research"})),
    ("Find external information comparing LangGraph and AutoGen.", frozenset({"research"})),
    ("Search online for today's weather in Beirut.", frozenset({"research"})),
    ("Research the history of PostgreSQL from public sources.", frozenset({"research"})),
    ("Look up Tesla Q4 revenue from public filings or news.", frozenset({"research"})),
    ("Find public articles about multi-agent RAG systems.", frozenset({"research"})),
    ("Research the latest iPhone release date.", frozenset({"research"})),
    ("Search the web for DuckDuckGo API documentation.", frozenset({"research"})),
    ("Look up who the current UN Secretary-General is.", frozenset({"research"})),
    ("Find external reviews of Groq LPU inference.", frozenset({"research"})),
    ("Research climate change statistics from public sources.", frozenset({"research"})),
    ("Look up the population of Tokyo from public data.", frozenset({"research"})),
    ("Find recent papers on vector databases online.", frozenset({"research"})),
    ("Search for the official Python 3.13 release notes online.", frozenset({"research"})),
    ("Research the current Bitcoin price on the public web.", frozenset({"research"})),
    # Freshness: current facts without an explicit "research" verb.
    ("Who is the current CEO of OpenAI?", frozenset({"research"})),
    ("Who is the current president of France?", frozenset({"research"})),
    ("What is the latest version of Python?", frozenset({"research"})),
]

VISU_CASES = [
    ("Make a bar chart of sales by region.", frozenset({"visu", "sql"})),
    ("Create a pie chart of employee salaries by department.", frozenset({"visu", "sql"})),
    ("Plot a line chart of sales over time.", frozenset({"visu", "sql"})),
    ("Visualize sales amounts as a bar graph.", frozenset({"visu", "sql"})),
    ("Draw a pie chart of employees per department.", frozenset({"visu", "sql"})),
    ("Chart the total sales by region.", frozenset({"visu", "sql"})),
    ("Generate a bar graph of salary per employee.", frozenset({"visu", "sql"})),
    ("Visualise the sales data as a chart.", frozenset({"visu", "sql"})),
    ("Create a line chart of monthly sales.", frozenset({"visu", "sql"})),
    ("Plot employee count by department as a bar chart.", frozenset({"visu", "sql"})),
    ("Make a graph of sales in MENA vs EU.", frozenset({"visu", "sql"})),
    ("Show a pie chart of regional sales share.", frozenset({"visu", "sql"})),
    ("Create a chart of amount by region.", frozenset({"visu", "sql"})),
    ("Draw a line graph of sale_date versus amount.", frozenset({"visu", "sql"})),
    ("Plot a pie chart of department headcount.", frozenset({"visu", "sql"})),
    ("Make a bar chart titled Sales by Region.", frozenset({"visu", "sql"})),
    ("Graph the salaries of all employees.", frozenset({"visu", "sql"})),
    ("Visualize employee distribution with a pie chart.", frozenset({"visu", "sql"})),
    ("Create a bar plot of sales totals.", frozenset({"visu", "sql"})),
    ("Please visualize this as a pie chart.", frozenset({"visu"})),
    ("Do a graph of employees and department. There are 3 departments X, Y, Z: 5 employees in X, 4 in Y, 3 in Z.",frozenset({"visu"}),),
    ("Make a pie chart: 60% of sales came from EU, 25% from MENA, 15% from APAC.",frozenset({"visu"}),),
    ("Chart this: Q1 revenue was 100k, Q2 was 150k, Q3 was 120k, Q4 was 200k.",frozenset({"visu"}),),
    ("Plot a bar chart with these values — apples: 12, oranges: 7, bananas: 9.",frozenset({"visu"}),),
    ("Visualize this data as a line chart: Jan 10, Feb 15, Mar 13, Apr 20.",frozenset({"visu"}),),
]

CONVO_CASES = [
    ("Hello", frozenset({"convo"})),
    ("Hi, what can you help with?", frozenset({"convo"})),
    ("What does SQL stand for?", frozenset({"convo"})),
    ("Thanks, that was helpful.", frozenset({"convo"})),
    ("Who are you?", frozenset({"convo"})),
    ("Good morning", frozenset({"convo"})),
    ("What is a database, in one sentence?", frozenset({"convo"})),
    ("How does this assistant work at a high level?", frozenset({"convo"})),
    ("Nice to meet you", frozenset({"convo"})),
    ("What does RAG mean?", frozenset({"convo"})),
    ("Just checking if you're online.", frozenset({"convo"})),
    ("Explain what a pie chart is.", frozenset({"convo"})),
    ("Hi there", frozenset({"convo"})),
    ("Thank you", frozenset({"convo"})),
    ("What specialists do you have, in simple terms?", frozenset({"convo"})),
    ("Hey, are you there?", frozenset({"convo"})),
    ("That's all for now.", frozenset({"convo"})),
]

CLARIFICATION_CASES = [
    ("Tell me about the numbers.", frozenset({"clarification"})),
    ("I need the data for the report.", frozenset({"clarification"})),
    ("Show me that.", frozenset({"clarification"})),
    ("The usual.", frozenset({"clarification"})),
    ("Can you get the info?", frozenset({"clarification"})),
    ("Check it.", frozenset({"clarification"})),
    ("Same as last time.", frozenset({"clarification"})),
    ("Do the thing.", frozenset({"clarification"})),
    ("Update it.", frozenset({"clarification"})),
    ("What about the other one?", frozenset({"clarification"})),
    ("I need those figures.", frozenset({"clarification"})),
    ("Handle this request.", frozenset({"clarification"})),
    ("The report.", frozenset({"clarification"})),
    ("Look into it.", frozenset({"clarification"})),
    ("Give me the details.", frozenset({"clarification"})),
    ("Process the request I mentioned.", frozenset({"clarification"})),
    ("You know what I mean.", frozenset({"clarification"})),
    ("Get me the stuff.", frozenset({"clarification"})),
    ("Continue.", frozenset({"clarification"})),
    ("Make it better.", frozenset({"clarification"})),
]


def _route(prompt: str) -> str:
    state = SupervisorState(
        messages=[HumanMessage(content=prompt)],
        turn_count=1,
    )
    context = gather_context(state, task_history=[], user_id=None)
    decision = get_supervisor_decision(context, llm())
    assert decision is not None, "supervisor routing returned no decision"
    return decision.next


def _plan_routes(prompt: str) -> list[str]:
    state = SupervisorState(
        messages=[HumanMessage(content=prompt)],
        turn_count=1,
    )
    context = gather_context(state, task_history=[], user_id=None)
    plan, _ = _validate_plan(get_workflow_plan(context, llm()), state)
    return [item.route for item in plan]


def _report(label, hits, total, min_accuracy, mistakes):
    accuracy = hits / total if total else 0.0
    print(f"[{label}] accuracy={accuracy:.0%} ({hits}/{total})  threshold={min_accuracy:.0%}")
    if accuracy < min_accuracy:
        detail = "\n".join(mistakes) or "(none)"
        pytest.fail(
            f"{label} routing accuracy {accuracy:.0%} ({hits}/{total}) "
            f"below {min_accuracy:.0%}\n{detail}"
        )


def _assert_plan_exact(cases, label, min_accuracy):
    """cases: (prompt, expected_route_list); the plan must match exactly."""
    mistakes = []
    hits = 0
    for prompt, expected in cases:
        plan = _plan_routes(prompt)
        ok = plan == expected
        print(
            f"[{label}] {'OK' if ok else 'MISS'}  expected={expected}  "
            f"got={plan!r}  {prompt!r}"
        )
        if ok:
            hits += 1
        else:
            mistakes.append(f"  got={plan!r} expected={expected}  {prompt!r}")
    _report(label, hits, len(cases), min_accuracy, mistakes)


def _assert_plan_set(cases, label, min_accuracy):
    """cases: (prompt, {acceptable_frozensets}); the plan's route set must match."""
    mistakes = []
    hits = 0
    for prompt, acceptable in cases:
        plan = _plan_routes(prompt)
        ok = frozenset(plan) in acceptable
        expected = sorted(sorted(group) for group in acceptable)
        print(
            f"[{label}] {'OK' if ok else 'MISS'}  expected={expected}  "
            f"got={plan!r}  {prompt!r}"
        )
        if ok:
            hits += 1
        else:
            mistakes.append(f"  got={plan!r} acceptable={acceptable}  {prompt!r}")
    _report(label, hits, len(cases), min_accuracy, mistakes)


# --- planner route precision (primary path) ---------------------------------


def test_llm_planner_routes_sql_prompts():
    _assert_plan_exact([(p, ["sql"]) for p, _ in SQL_CASES], "sql", MIN_ACCURACY["sql"])


def test_llm_planner_routes_rag_prompts():
    _assert_plan_exact([(p, ["rag"]) for p, _ in RAG_CASES], "rag", MIN_ACCURACY["rag"])


def test_llm_planner_routes_research_prompts():
    _assert_plan_exact(
        [(p, ["research"]) for p, _ in RESEARCH_CASES],
        "research",
        MIN_ACCURACY["research"],
    )


def test_llm_planner_routes_convo_prompts():
    _assert_plan_exact(
        [(p, ["convo"]) for p, _ in CONVO_CASES], "convo", MIN_ACCURACY["convo"]
    )


def test_llm_planner_routes_clarification_prompts():
    _assert_plan_exact(
        [(p, ["clarification"]) for p, _ in CLARIFICATION_CASES],
        "clarification",
        MIN_ACCURACY["clarification"],
    )


# VISU_CASES layout: first 19 are charts over database data (sql then visu),
# index 19 is a chart with no data at all, the last 5 carry inline values.
VISU_PLAN_CASES = (
    [(prompt, {frozenset({"sql", "visu"})}) for prompt, _ in VISU_CASES[:19]]
    + [
        (prompt, {frozenset({"visu"}), frozenset({"clarification"})})
        for prompt, _ in VISU_CASES[19:20]
    ]
    + [(prompt, {frozenset({"visu"})}) for prompt, _ in VISU_CASES[20:]]
)


def test_llm_planner_routes_visu_prompts():
    _assert_plan_set(VISU_PLAN_CASES, "visu", MIN_ACCURACY["visu"])


# --- fallback smoke (planning-schema-failure path) --------------------------


FALLBACK_SMOKE = [
    ("How many employees are there?", {"sql"}),
    ("What does our remote work policy say?", {"rag"}),
    ("Who is the current CEO of OpenAI?", {"research"}),
    ("Hello", {"convo"}),
    ("Tell me about the numbers.", {"clarification"}),
    ("Show sales by region as a pie chart.", {"visu", "sql"}),
]


def test_llm_fallback_routes_smoke():
    mistakes = []
    for prompt, accepted in FALLBACK_SMOKE:
        chosen = _route(prompt)
        ok = chosen in accepted
        print(
            f"[fallback] {'OK' if ok else 'MISS'}  expected={sorted(accepted)}  "
            f"got={chosen!r}  {prompt!r}"
        )
        if not ok:
            mistakes.append(f"  got={chosen!r} expected={sorted(accepted)}  {prompt!r}")
    if mistakes:
        pytest.fail("fallback routing smoke failed:\n" + "\n".join(mistakes))


PLANNER_CASES = [
    # Single-intent requests.
    ("How many employees are there?", ["sql"]),
    ("What does our remote work policy say?", ["rag"]),
    ("Who is the current CEO of OpenAI?", ["research"]),
    ("Hello, how are you?", ["convo"]),
    ("Tell me about the numbers.", ["clarification"]),
    # Two distinct asks, in the order they should run.
    (
        "How many employees are there, and what does the remote work policy say?",
        ["sql", "rag"],
    ),
    ("Show sales by region as a pie chart.", ["sql", "visu"]),
    (
        "How many employees do we have, and what is the current price of gold?",
        ["sql", "research"],
    ),
    (
        "Who is the current CEO of Tesla, and how many employees do we have?",
        ["research", "sql"],
    ),
    # Inline chart data needs no sql step.
    ("Make a pie chart: 60% EU, 25% MENA, 15% APAC.", ["visu"]),
    # Three distinct asks.
    (
        "How many employees are there, who is the current CEO of OpenAI, "
        "and what does our remote work policy say?",
        ["sql", "research", "rag"],
    ),
    (
        "Show sales by region as a pie chart, and what does the remote work policy say?",
        ["sql", "visu", "rag"],
    ),
    (
        "Who is the current CEO of Tesla, how many employees do we have, "
        "and make a pie chart: 60% EU, 40% US.",
        ["research", "sql", "visu"],
    ),
]


@pytest.mark.parametrize("prompt,expected", PLANNER_CASES)
def test_llm_planner_decomposes_workflows(prompt, expected):
    assert _plan_routes(prompt) == expected


# Order-ambiguous pairs: the planner may reasonably put the database count
# before or after the document lookup, so assert the set, not the order.
PLANNER_UNORDERED_CASES = [
    (
        "What does our remote work policy say, and how many employees are there?",
        {"rag", "sql"},
    ),
    (
        "Summarize our lessons on state design, and count the sales rows.",
        {"rag", "sql"},
    ),
]


@pytest.mark.parametrize("prompt,expected", PLANNER_UNORDERED_CASES)
def test_llm_planner_decomposes_unordered_workflows(prompt, expected):
    assert set(_plan_routes(prompt)) == expected


def test_llm_planner_handles_greeting_plus_tasks():
    """A greeting may or may not become its own step; the tasks must appear in
    order regardless."""
    routes = _plan_routes(
        "Hi! Who is the current CEO of OpenAI, and how many employees do we have?"
    )
    assert [route for route in routes if route != "convo"] == ["research", "sql"]


def test_llm_planner_caps_long_requests():
    routes = _plan_routes(
        "How many employees are there, what does the policy say, "
        "who is the current CEO of OpenAI, and chart the sales by region."
    )
    assert len(routes) <= MAX_PLAN_STEPS
    assert all(
        route in {"rag", "convo", "sql", "research", "visu", "clarification"}
        for route in routes
    )