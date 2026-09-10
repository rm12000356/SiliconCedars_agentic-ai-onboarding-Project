"""
LLM routing-precision tests for the supervisor residual path.

First-turn requests (no last_result) fall through to the real model.
Each route has ~20 prompts.

Run:  pytest -m llm tests/test_llm_supervisor_routing.py -v -s
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

import pytest

from agents.supervisor import gather_context, get_supervisor_decision
from services.llm import llm
from state.state import SupervisorState

from tests.conftest import requires_llm


pytestmark = [pytest.mark.llm, requires_llm]

MIN_ACCURACY = {
    "sql": 0.70,
    "rag": 0.70,
    "research": 0.70,
    "visu": 0.60,
    "convo": 0.60,
    "clarification": 0.30,
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
]

CONVO_CASES = [
    ("Hello", frozenset({"convo"})),
    ("Hi, what can you help with?", frozenset({"convo"})),
    ("What does SQL stand for?", frozenset({"convo"})),
    ("Thanks, that was helpful.", frozenset({"convo"})),
    ("Who are you?", frozenset({"convo"})),
    ("Good morning", frozenset({"convo"})),
    ("Can you rephrase that more simply?", frozenset({"convo"})),
    ("What is a database, in one sentence?", frozenset({"convo"})),
    ("How does this assistant work at a high level?", frozenset({"convo"})),
    ("Nice to meet you", frozenset({"convo"})),
    ("What does RAG mean?", frozenset({"convo"})),
    ("Just checking if you're online.", frozenset({"convo"})),
    ("Explain what a pie chart is.", frozenset({"convo"})),
    ("Please summarize that more briefly.", frozenset({"convo"})),
    ("Hi there", frozenset({"convo"})),
    ("Thank you", frozenset({"convo"})),
    ("Can you say that again?", frozenset({"convo"})),
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
    return decision.next


def _assert_precision(cases, label, min_accuracy):
    mistakes = []
    hits = 0
    for prompt, accepted in cases:
        chosen = _route(prompt)
        ok = chosen in accepted
        mark = "OK" if ok else "MISS"
        print(f"[{label}] {mark}  expected={sorted(accepted)}  got={chosen!r}  {prompt!r}")
        if ok:
            hits += 1
        else:
            mistakes.append(f"  got={chosen!r} expected={sorted(accepted)}  {prompt!r}")

    total = len(cases)
    accuracy = hits / total if total else 0.0
    print(f"[{label}] accuracy={accuracy:.0%} ({hits}/{total})  threshold={min_accuracy:.0%}")
    if accuracy < min_accuracy:
        detail = "\n".join(mistakes) or "(none)"
        pytest.fail(
            f"{label} routing accuracy {accuracy:.0%} ({hits}/{total}) "
            f"below {min_accuracy:.0%}\n{detail}"
        )


def test_llm_routes_sql_prompts():
    _assert_precision(SQL_CASES, "sql", MIN_ACCURACY["sql"])


def test_llm_routes_rag_prompts():
    _assert_precision(RAG_CASES, "rag", MIN_ACCURACY["rag"])


def test_llm_routes_research_prompts():
    _assert_precision(RESEARCH_CASES, "research", MIN_ACCURACY["research"])


def test_llm_routes_visu_prompts():
    _assert_precision(VISU_CASES, "visu", MIN_ACCURACY["visu"])


def test_llm_routes_convo_prompts():
    _assert_precision(CONVO_CASES, "convo", MIN_ACCURACY["convo"])


def test_llm_routes_clarification_prompts():
    _assert_precision(CLARIFICATION_CASES, "clarification", MIN_ACCURACY["clarification"])