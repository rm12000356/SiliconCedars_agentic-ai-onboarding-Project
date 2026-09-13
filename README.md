# SiliconCedars Agentic AI Onboarding Project

## Multi-Agent Supervisor

A hierarchical AI assistant built with LangGraph. A main supervisor routes requests to specialized agents for SQL, web research, RAG, visualization, and conversation.

Deterministic code guards, including attempt counters, task history, `done → end`, and same-turn SQL → visualization, handle loop prevention and routing invariants that proved unreliable when left purely to prompt instructions.

The LLM is only consulted for genuinely novel routing decisions.

## Structure

* `agents/`
  Main supervisor and specialized agents (SQL, visualization, conversation, RAG), plus the `research/` subgraph (sub-supervisor, researcher, report writer).

* `graph/`
  LangGraph orchestration. `workflow.py` builds the graph, while `routing.py` centralizes routing functions.

* `state/`
  Shared state definitions, including `SupervisorState`, `SubGraphSupervisorState`, `SpecialistResult`, `TaskRecord`, and related types.

* `tools/`
  Database access (`database.py`) and web search/fetch (`web_search.py`), including SSRF protections.

* `rag/`
  Indexing and retrieval over `lessons_learned`, backed by pgvector.

* `services/`

  * `llm.py`: Groq with OpenRouter fallback
  * `memory.py`: Checkpointer and long-term user memory
  * `errors.py`: Authentication vs. transient LLM error classification

* `evaluation/`
  LangSmith evaluation for routing via `task_history`, RAG grounding, and correctness.

* `config/`
  Reserved for centralized settings. Not yet populated.

* `db/`
  Docker/Postgres initialization, including schema, roles, and pgvector.

* `tests/`
  Unit, specialist, LLM, and integration tests.

* `main.py`
  CLI entry point.

* `chat.py`
  Chainlit web chat entry point.

## Architecture Notes

### Security

Sensitive data such as salaries and credentials is protected by two layers:

1. Which tools are bound through `permission_level` in the configuration.
2. PostgreSQL role grants. `general_role` has no grants on `salaries` or `credentials`.

Keyword checks in `Sql_agent` are a convenience fast-path, not the security boundary.

### Deterministic Guards

The system uses deterministic guards rather than relying entirely on the model:

* Never loop on a failed specialist.
* Stop after a defined number of research attempts.
* Force `end` after a successful result.
* Automatically route SQL → visualization in the same turn when `structured_data` exists and the user asked for a chart.

These rules are enforced in `deterministic_decision`.

### Long-Term Memory

Long-term memory is keyed by `user_id` and stores flat key-value facts in PostgreSQL.

A keyword heuristic gates whether an extraction LLM call is made.

### Short-Term Memory

`memory_manager` runs once per turn.

It:

* Summarizes old messages after they pass a length threshold.
* Prunes old `task_history`.
* Keeps the current-turn history available because routing guards only read the current turn.

## Setup

### 1. Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
GROQ_API_KEY=              # required
LANGSMITH_API_KEY=         # optional, tracing and evaluation
OPENROUTER_API_KEY=        # optional, fallback LLM provider
DATABASE_URL=              # required if CHECKPOINT_BACKEND=postgres
CHAINLIT_DATABASE_URL=     # required for `chainlit run chat.py`

DB_HOST=localhost
DB_PORT=5432
DB_NAME=company_intel
DB_GENERAL_USER=general_role
DB_GENERAL_PASSWORD=general_dev_password
DB_ELEVATED_USER=app_owner
DB_ELEVATED_PASSWORD=devpassword
```

Optional variables with defaults:

```env
DEFAULT_USER_ID=test-user-1
DEFAULT_THREAD_ID=thread-test-user-1
DEFAULT_PERMISSION_LEVEL=elevated   # "general" or "elevated"
CHECKPOINT_BACKEND=memory            # "memory" or "postgres"
```

Use `general` unless you are explicitly testing salary or credential paths.

### 2. Database

Start PostgreSQL with:

```bash
docker compose up -d
```

This starts Postgres with pgvector and runs `db/init/` on first startup. Initialization includes the schema, seed data, roles, and `general_embeddings`.

If you change the Docker image or initialization scripts after the volume already exists:

```bash
docker compose down -v
docker compose up -d
```

Check the installed extensions:

```bash
docker exec -it silicon_cedars_db psql -U app_owner -d company_intel -c "\dx"
```

Check the tables:

```bash
docker exec -it silicon_cedars_db psql -U app_owner -d company_intel -c "\dt"
```

Expected tables include:

* `employees`
* `sales`
* `salaries`
* `credentials`
* `lessons_learned`
* `general_embeddings`

The `vector` extension should also be installed.

### 3. Python Dependencies

Create and activate the virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Unix
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

This is a frozen environment snapshot. `pywin32` is skipped on non-Windows systems.

### 4. Index the RAG Corpus

Run this once, and again whenever `lessons_learned` changes:

```bash
python -m rag.indexing
```

## Running

### CLI

```bash
python main.py
```

Type `quit` or `exit` to leave.

### Web

```bash
chainlit run chat.py
```

This requires `CHAINLIT_DATABASE_URL`.

## Testing

Run the unit tests:

```bash
pytest
```

Run live LLM tests:

```bash
pytest -m llm
```

Run integration tests:

```bash
pytest -m integration
```

### Test Categories

* **Unit**
  Routing, `map_to_state`, research controller, supervisor rules, error classification, and graph compilation.

* **Specialist / LLM**
  `test_llm_*.py`, SQL permission gating, and RAG grounding.

* **Security**
  `test_security_boundaries.py` and `test_sql_roles.py`. The database role is the actual enforcement point.

## Evaluation

Evaluation requires `LANGSMITH_API_KEY`.

Dataset names are created once. Bump the name, for example `routing-eval-v2`, if the examples change.

```python
from graph.workflow import Main_WorkFlow
from services.memory import get_checkpointer
from evaluation.evaluation import (
    run_routing_evaluation,
    run_rag_evaluation,
)

with get_checkpointer() as memory:
    graph = Main_WorkFlow(memory)

    run_routing_evaluation("routing-eval-v1", graph)
    run_rag_evaluation("rag-eval-v1")
```

### Routing Evaluation

Routing evaluation uses `task_history` to determine which specialist ran.

Clarification is scored through the graph interrupt because it is not recorded in `task_history`.

Chart requests may appear as SQL followed by visualization in the same turn.

### RAG Evaluation

RAG evaluation checks structural status and issue information for out-of-corpus honesty.

LLM-as-judge correctness is used only when a reference answer exists.

## Known Limitations

* **No real authentication**
  `user_id` and `permission_level` come from the entry-point configuration rather than a verified identity.

* **No cross-turn data chaining**
  Requests such as `"chart that"` after a previous `Finalize` are not supported. Same-turn SQL → visualization is supported.

