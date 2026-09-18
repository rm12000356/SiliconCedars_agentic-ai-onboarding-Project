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
  * `chart_storage.py`: Local storage client and authenticated chart serving
  * `logging_config.py`: LOG_LEVEL-driven logging setup

* `evaluation/`
  LangSmith evaluation for routing via `task_history`, RAG grounding, and correctness.

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
DEFAULT_PERMISSION_LEVEL=general    # "general" or "elevated"
CHECKPOINT_BACKEND=memory            # "memory" or "postgres"
LOG_LEVEL=INFO                       # DEBUG, INFO, WARNING, ERROR
DB_POOL_MIN_SIZE=1
DB_POOL_MAX_SIZE_GENERAL=5
DB_POOL_MAX_SIZE_ELEVATED=3
DB_POOL_ACQUIRE_TIMEOUT=30
DB_CONNECT_TIMEOUT=5
CHART_STORAGE_DIR=.chainlit_charts
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

### Prerequisites

`pytest.ini` sets a default filter (`-m "not llm and not integration"`), so a plain `pytest` run never calls the LLM or touches the database.

* **LLM tests** (`llm`): require `GROQ_API_KEY` or `OPENROUTER_API_KEY`.
* **Integration tests** (`integration`): require a running Postgres (`docker compose up -d`) and the `DB_*` / `DATABASE_URL` environment variables.
* **Postgres checkpointer tests**: additionally require `DATABASE_URL` and `CHECKPOINT_BACKEND=postgres`.

### Test Commands

| Command | What it runs |
| --- | --- |
| `pytest` | Unit tests only (default; excludes `llm` and `integration`) |
| `pytest -m "not llm and not integration"` | Same as `pytest`, written explicitly |
| `pytest -m llm` | Live LLM tests (`test_llm_*.py`, RAG grounding, SQL permission gating) |
| `pytest -m integration` | Tests requiring live services (DB roles, checkpointer, pools, chart route) |
| `pytest -m "llm or integration"` | All live tests |
| `pytest -m "not llm"` | Unit + integration |
| `pytest -m "not integration"` | Unit + LLM |
| `pytest tests/test_routing.py -m` | A single test file |
| `pytest tests/test_routing.py::test_name -m` | A single test |
| `pytest -k "sql"` | Tests whose name matches a keyword |
| `pytest tests/test_security_boundaries.py tests/test_sql_roles.py` | Security keyword boundaries + DB role boundaries |
| `pytest --markers` | List registered markers |
| `pytest --collect-only -q` | List tests without running them |
| `pytest -v` | Verbose per-test output |
| `pytest -x` | Stop on the first failure |

To exclude a marker, quote the expression: `pytest -m "not llm"`. The form `pytest -m -llm` is not valid.

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

memory, _ = get_checkpointer()
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

* **CLI has no real authentication**
  The `main.py` CLI reads `user_id` and `permission_level` from its configuration rather than a verified identity. The Chainlit entry point (`chat.py`) authenticates against `app_users` with bcrypt and derives `permission_level` from the logged-in user.

* **Charts in the web UI**
  The Chainlit entry point renders the generated PNG inline (`cl.Image`). Element
  files are persisted by a local storage client and served through an
  authenticated `/charts/<token>` route, so charts survive a page refresh. The
  `main.py` CLI prints the file path instead of rendering the image.

* **No cross-turn data chaining**
  Requests such as `"chart that"` after a previous `Finalize` are not supported. Same-turn SQL → visualization is supported.

