# Routing / LLM evaluation baseline

- **Date:** 2026-09-22
- **Commit:** `ae8b0370edaa3c9a1b9a211175f9268fc9753176` (branch `fix/routing-and-guards`)
- **Model:** Groq `openai/gpt-oss-120b` (primary)
- **Command:** `pytest -m llm tests/test_llm_supervisor_routing.py`
- **Result:** all routing categories at 100% accuracy, threshold 90%

Raw output follows.

---

tests/test_llm_supervisor_routing.py::test_llm_planner_routes_sql_prompts [sql] OK  expected=['sql']  got=['sql']  'How many employees are there?'
[sql] OK  expected=['sql']  got=['sql']  'How many sales records are in the database?'
[sql] OK  expected=['sql']  got=['sql']  'Show me the total salary of every employee.'
[sql] OK  expected=['sql']  got=['sql']  "What is Alice Example's department?"
[sql] OK  expected=['sql']  got=['sql']  'List all employees in Engineering.'
[sql] OK  expected=['sql']  got=['sql']  'What is the total sales amount?'
[sql] OK  expected=['sql']  got=['sql']  'Show sales totals grouped by region.'
[sql] OK  expected=['sql']  got=['sql']  'Find the employee ID for Rami Noueihed.'
[sql] OK  expected=['sql']  got=['sql']  'How many sales happened in the MENA region?'
[sql] OK  expected=['sql']  got=['sql']  "What is Rami Noueihed's salary?"
[sql] OK  expected=['sql']  got=['sql']  'Count employees per department.'
[sql] OK  expected=['sql']  got=['sql']  'What is the latest sale_date in the sales table?'
[sql] OK  expected=['sql']  got=['sql']  'Sum of sales in the EU region.'
[sql] OK  expected=['sql']  got=['sql']  'Which employees are in the Sales department?'
[sql] OK  expected=['sql']  got=['sql']  'Return every row from the employees table.'
[sql] OK  expected=['sql']  got=['sql']  'What is the average sale amount?'
[sql] OK  expected=['sql']  got=['sql']  'How many rows are in lessons_learned?'
[sql] OK  expected=['sql']  got=['sql']  'Show employee names and departments.'
[sql] OK  expected=['sql']  got=['sql']  'Get the credential record for user id 1.'
[sql] OK  expected=['sql']  got=['sql']  'List sales with amount greater than 1000.'
[sql] accuracy=100% (20/20)  threshold=90%
PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_routes_rag_prompts [rag] OK  expected=['rag']  got=['rag']  'What is the company remote work policy?'
[rag] OK  expected=['rag']  got=['rag']  'Explain the internal onboarding process for new hires.'
[rag] OK  expected=['rag']  got=['rag']  'What did we learn about SQL security from past internal projects?'
[rag] OK  expected=['rag']  got=['rag']  'What did we learn internally about preventing agent routing loops?'
[rag] OK  expected=['rag']  got=['rag']  'Summarize our internal lessons on LangGraph state design.'
[rag] OK  expected=['rag']  got=['rag']  'What is our internal policy on database access?'
[rag] OK  expected=['rag']  got=['rag']  'How should we split SQL permissions according to internal docs?'
[rag] OK  expected=['rag']  got=['rag']  'What did the Internal Onboarding Bot v1 project teach us?'
[rag] OK  expected=['rag']  got=['rag']  'Company procedure for handling sensitive credentials.'
[rag] OK  expected=['rag']  got=['rag']  'What are our internal architecture guidelines for multi-agent systems?'
[rag] OK  expected=['rag']  got=['rag']  'Explain our internal lessons about RAG fallback behavior.'
[rag] OK  expected=['rag']  got=['rag']  'What does internal documentation say about turn counters and memory?'
[rag] OK  expected=['rag']  got=['rag']  'Internal process notes from the Customer Support Assistant project.'
[rag] OK  expected=['rag']  got=['rag']  'What organizational procedures exist for newemployee onboarding?'
[rag] OK  expected=['rag']  got=['rag']  'Company policy on who can query salary data,from internal docs.'
[rag] OK  expected=['rag']  got=['rag']  'What did we learn from the Customer Support Assistant internally?'
[rag] OK  expected=['rag']  got=['rag']  'According to internal guidance, should the model decide authorization?'
[rag] OK  expected=['rag']  got=['rag']  'What tags do we use on internal lessons_learned documents?'
[rag] OK  expected=['rag']  got=['rag']  'Describe the internal lesson about prompt-only loop prevention.'
[rag] OK  expected=['rag']  got=['rag']  'What is the official internal guidance on supervisor design?'
[rag] accuracy=100% (20/20)  threshold=90%
PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_routes_research_prompts [research] OK  expected=['research']  got=['research']  'Research the latest market share of Tesla in 2025.'
[research] OK  expected=['research']  got=['research']  "Find recent news about OpenAI's latest model releases."
[research] OK  expected=['research']  got=['research']  'Look up the capital of Franceon the public web.'
[research] OK  expected=['research']  got=['research']  'Search the web for the current price of gold.'
[research] OK  expected=['research']  got=['research']  'Research who invented the Python programming language.'
[research] OK  expected=['research']  got=['research']  'Look up recent public news about NVIDIA GPUs.'
[research] OK  expected=['research']  got=['research']  'Find external information comparing LangGraph and AutoGen.'
[research] OK  expected=['research']  got=['research']  "Search online for today's weather in Beirut."
[research] OK  expected=['research']  got=['research']  'Research the history of PostgreSQL from public sources.'
[research] OK  expected=['research']  got=['research']  'Look up Tesla Q4 revenue frompublic filings or news.'
[research] OK  expected=['research']  got=['research']  'Find public articles about multi-agent RAG systems.'
[research] OK  expected=['research']  got=['research']  'Research the latest iPhone release date.'
[research] OK  expected=['research']  got=['research']  'Search the web for DuckDuckGoAPI documentation.'
[research] OK  expected=['research']  got=['research']  'Look up who the current UN Secretary-General is.'
[research] OK  expected=['research']  got=['research']  'Find external reviews of GroqLPU inference.'
[research] OK  expected=['research']  got=['research']  'Research climate change statistics from public sources.'
[research] OK  expected=['research']  got=['research']  'Look up the population of Tokyo from public data.'
[research] OK  expected=['research']  got=['research']  'Find recent papers on vector databases online.'
[research] OK  expected=['research']  got=['research']  'Search for the official Python 3.13 release notes online.'
[research] OK  expected=['research']  got=['research']  'Research the current Bitcoin price on the public web.'
[research] OK  expected=['research']  got=['research']  'Who is the current CEO of OpenAI?'
[research] OK  expected=['research']  got=['research']  'Who is the current president of France?'
[research] OK  expected=['research']  got=['research']  'What is the latest version ofPython?'
[research] accuracy=100% (23/23)  threshold=90%
PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_routes_convo_prompts [convo] OK expected=['convo']  got=['convo']  'Hello'
[convo] OK  expected=['convo']  got=['convo']  'Hi, what can you help with?'
[convo] OK  expected=['convo']  got=['convo']  'What does SQL stand for?'
[convo] OK  expected=['convo']  got=['convo']  'Thanks, that was helpful.'
[convo] OK  expected=['convo']  got=['convo']  'Who are you?'
[convo] OK  expected=['convo']  got=['convo']  'Good morning'
[convo] OK  expected=['convo']  got=['convo']  'What is a database, in one sentence?'
[convo] OK  expected=['convo']  got=['convo']  'How does this assistant work at a highlevel?'
[convo] OK  expected=['convo']  got=['convo']  'Nice to meet you'
[convo] OK  expected=['convo']  got=['convo']  'What does RAG mean?'
[convo] OK  expected=['convo']  got=['convo']  "Just checking if you're online."
[convo] OK  expected=['convo']  got=['convo']  'Explain what a pie chart is.'
[convo] OK  expected=['convo']  got=['convo']  'Hi there'
[convo] OK  expected=['convo']  got=['convo']  'Thank you'
[convo] OK  expected=['convo']  got=['convo']  'What specialists do you have, in simple terms?'
[convo] OK  expected=['convo']  got=['convo']  'Hey, are you there?'
[convo] OK  expected=['convo']  got=['convo']  "That's all for now."
[convo] accuracy=100% (17/17)  threshold=90%
PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_routes_clarification_prompts [clarification] OK  expected=['clarification']  got=['clarification']  'Tell me about thenumbers.'
[clarification] OK  expected=['clarification']  got=['clarification']  'I need the data for the report.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Show me that.'
[clarification] OK  expected=['clarification']  got=['clarification']  'The usual.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Can you get the info?'
[clarification] OK  expected=['clarification']  got=['clarification']  'Check it.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Same as last time.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Do the thing.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Update it.'
[clarification] OK  expected=['clarification']  got=['clarification']  'What about theother one?'
[clarification] OK  expected=['clarification']  got=['clarification']  'I need those figures.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Handle this request.'
[clarification] OK  expected=['clarification']  got=['clarification']  'The report.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Look into it.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Give me the details.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Process the request I mentioned.'
[clarification] OK  expected=['clarification']  got=['clarification']  'You know what I mean.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Get me the stuff.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Continue.'
[clarification] OK  expected=['clarification']  got=['clarification']  'Make it better.'
[clarification] accuracy=100% (20/20)  threshold=90%
PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_routes_visu_prompts [visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Make a bar chart of sales by region.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Create a pie chart of employee salaries by department.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Plot a line chart of sales over time.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Visualize sales amounts as a bar graph.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Draw a pie chart of employees per department.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Chart the total sales by region.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Generate a bar graph of salary per employee.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Visualise the sales data as a chart.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Create a line chart of monthly sales.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Plot employee count by department as a bar chart.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Make a graph of sales in MENA vs EU.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Show a pie chart of regional sales share.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Create a chart of amount by region.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Draw a line graph of sale_date versus amount.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Plot a pie chart of department headcount.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Make a bar chart titled Sales by Region.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Graph the salaries of allemployees.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Visualize employee distribution with a pie chart.'
[visu] OK  expected=[['sql', 'visu']]  got=['sql', 'visu']  'Create a bar plot of sales totals.'
[visu] OK  expected=[['clarification'], ['visu']]  got=['clarification']  'Please visualize this as a pie chart.'
[visu] OK  expected=[['visu']]  got=['visu']  'Do a graph of employees and department.There are 3 departments X, Y, Z: 5 employees in X, 4 in Y, 3 in Z.'
[visu] OK  expected=[['visu']]  got=['visu']  'Make a pie chart: 60% of sales came from EU, 25% from MENA, 15% from APAC.'
[visu] OK  expected=[['visu']]  got=['visu']  'Chart this: Q1 revenue was 100k, Q2 was150k, Q3 was 120k, Q4 was 200k.'
[visu] OK  expected=[['visu']]  got=['visu']  'Plot a bar chart with these values — apples: 12, oranges: 7, bananas: 9.'
[visu] OK  expected=[['visu']]  got=['visu']  'Visualize this data as a line chart: Jan 10, Feb 15, Mar 13, Apr 20.'
[visu] accuracy=100% (25/25)  threshold=90%
PASSED
tests/test_llm_supervisor_routing.py::test_llm_fallback_routes_smoke [fallback] OK  expected=['sql']  got='sql'  'How many employees are there?'
[fallback] OK  expected=['rag']  got='rag'  'What does our remote work policy say?'
[fallback] OK  expected=['research']  got='research'  'Who is the current CEO of OpenAI?'
[fallback] OK  expected=['convo']  got='convo'  'Hello'
[fallback] OK  expected=['clarification']  got='clarification'  'Tell me about the numbers.'
[fallback] OK  expected=['sql', 'visu']  got='sql'  'Show sales by region as a pie chart.'
PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[How many employees are there?-expected0] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[What does our remote work policy say?-expected1] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[Who is thecurrent CEO of OpenAI?-expected2] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[Hello, howare you?-expected3] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[Tell me about the numbers.-expected4] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[How many employees are there, and what does the remote work policy say?-expected5] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[Show salesby region as a pie chart.-expected6] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[How many employees do we have, and what is the current price of gold?-expected7] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[Who is thecurrent CEO of Tesla, and how many employees do we have?-expected8] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[Make a piechart: 60% EU, 25% MENA, 15% APAC.-expected9] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[How many employees are there, who is the current CEO of OpenAI, and what does our remote work policy say?-expected10] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[Show salesby region as a pie chart, and what does the remote work policy say?-expected11] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_workflows[Who is thecurrent CEO of Tesla, how many employees do we have, and chart the sales by region.-expected12] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_unordered_workflows[What does our remote work policy say, and how many employees are there?-expected0] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_decomposes_unordered_workflows[Summarize our lessons on state design, and count the sales rows.-expected1] PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_handles_greeting_plus_tasks PASSED
tests/test_llm_supervisor_routing.py::test_llm_planner_caps_long_requests PASSED