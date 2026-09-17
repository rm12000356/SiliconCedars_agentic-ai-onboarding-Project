# QA & Regression Report — SiliconCedars_agentic-ai-onboarding-Project

- **Date:** 2026-09-17
- **Branch:** `fix/clarification-and-routing-guards` (not `main`)
- **Base commit at start of review:** `59b42b3` — *"fix: preserve user intent across clarification and tighten routing guards"*
- **Tree state at start:** clean and up to date with origin.
- **Tree state during review:** **became dirty mid-session.** Files were edited externally (not by this QA pass) while I was testing. See "Working-tree changes observed mid-review" below. All findings below were re-verified against the current on-disk state.
- **Report mode:** review-only. No source or test files were modified by this pass. This file is the only artifact written.

## Environment actually used

| Resource | Status |
|---|---|
| Python | 3.13.15 in existing `.venv` |
| Dependencies | Every pin already installed and matching the (UTF-16LE) `requirements.txt`; no reinstall performed |
| Postgres/pgvector | **Real** — started `docker compose up -d postgres` (`pgvector/pgvector:pg16`, container `silicon_cedars_db`). 17 tables, seed data present, `general_role` least-privilege confirmed |
| Embedder | **Real** — `sentence-transformers/all-MiniLM-L6-v2` loaded from cache; 9 lessons indexed into `general_embeddings` |
| Checkpointers | Both exercised for real: `MemorySaver` and `PostgresSaver` (with the project's configured `JsonPlusSerializer`) |
| LLM calls | **Always faked.** No live model call was made. A scripted fake was installed at the point of use in every agent module |
| Live outbound web | Not exercised. Only the SSRF validator and a fake HTTP transport were driven |

`pytest.ini` deselects `llm`/`integration` by default, so the default suite exercises none of the LLM-facing logic. Those paths were driven manually with the fake harness.

## Working-tree changes observed mid-review

The tree was clean at `59b42b3`, then these **external, unstaged** edits appeared (`git diff`):

- `tests/test_supervisor_rules.py` — replaced the impossible-shaped `test_second_clarification_in_same_turn_ends` with `test_first_clarification_in_same_turn_is_allowed` and `test_clarification_cap_routes_to_convo`; added `clarification_count` to the `_state` helper.
- `agents/finalize.py` — moved `_already_delivered` above `Finalize`, removed the redundant `if not _last_is_ai_message(...)` branch, tidied imports/whitespace.
- `agents/memory_manager.py` — removed an unused `SystemMessage` import.
- `db/connection.py` — added an optional `connect_timeout` parameter.
- `tests/conftest.py` — probe now uses `get_general_connection(connect_timeout=2)`.
- `README.md` — checkpoint usage snippet corrected to the tuple API.

These resolve the severity-1 issue reported below. I verified them but did not author them.

---

## Baseline test counts (before/after the external test edit)

| Run | Result |
|---|---|
| `pytest` (default, base commit `59b42b3`, clean tree) | **1 failed, 61 passed, 29 deselected, 1 xfailed** (92 collected) |
| `pytest` (default, current tree after external edits) | **63 passed, 29 deselected, 1 xfailed** (93 collected, 0 failed) |
| Non-LLM integration subset (current tree) | `test_sql_roles.py` 2 passed; `test_security_boundaries.py -m "integration and not llm"` 3 passed |
| `llm`-marked suites | **Not run** (would call a live model; forbidden by the task) |

The single pre-edit failure, exactly:

```
_________________ test_second_clarification_in_same_turn_ends _________________
>       assert guarded.next == "end"
E       AssertionError: assert 'clarification' == 'end'
tests\test_supervisor_rules.py:243: AssertionError
1 failed, 61 passed, 29 deselected, 1 xfailed in 1.60s
```

---

## Findings, ranked by severity

### S1 — Stale regression test asserted a state shape production cannot produce (RESOLVED externally during review)

- **File/lines:** `tests/test_supervisor_rules.py:229-243` (original)
- **Why it was wrong:** the test built a `TaskRecord` with `route="clarification"` via `TaskRecord.model_construct(...)`. `TaskRecord.route` is typed `SpecialistRoute`, which structurally excludes `"clarification"`; and the production code never records clarification in `task_history`. The test then asserted the *old* task_history-based cap, while `post_decision_guards` (`agents/supervisor.py:286-303`) now reads `state.clarification_count`. `_state` left that at its default `0`, so the guard correctly returned the original decision and the assertion failed.
- **This is exactly the §11 `model_construct` red flag:** an assertion against a state shape real code cannot produce.
- **Real reproduction (before the external edit):** see baseline transcript above (`assert 'clarification' == 'end'`).
- **Status: RESOLVED.** The external edit replaced it with two valid tests that set `clarification_count` directly and assert the real semantics (first allowed, capped → `convo`). Current run: both pass. The old test would have been wrong to "fix" by widening the literal.
- **Regression evidence:** baseline `1 failed, 61 passed` → current `63 passed, 0 failed`.

### S2 — HIGH: non-ASCII `→` inside `print()` crashes the LLM fallback path and research nodes on a default Windows console (cp1252)

- **Files/lines (all U+2192):**
  - `services/llm.py:112` — `print(f"[LLM] {label} failed (...) → trying next model")`
  - `agents/research/supervisor.py:103` — `print("[SUB-SUPERVISOR] substantial research note already present → forcing report")`
  - `agents/research/researcher.py:172` and `:176`
- **Impact:** on this machine `sys.stdout.encoding` is **cp1252** (verified). Any retryable model failure makes the *logging itself* raise `UnicodeEncodeError` from inside `ResilientLLM.invoke`, so the designed model-fallback never happens; the node crashes instead. The same applies to the two research `print()` sites. This is the most likely-to-be-hit runtime defect found.
- **Real reproduction** (no `PYTHONIOENCODING` set, i.e. production default):

```
stdout encoding: cp1252
[LLM] Trying grok
RAISED: UnicodeEncodeError 'charmap' codec can't encode character '\u2192' in position 36
```

  Also reproduced organically during the research sweep:

```
File "...\agents\research\supervisor.py", line 103, in Sub_controler
    print("[SUB-SUPERVISOR] substantial research note already present \u2192 forcing report")
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 59
```

- **Proposed minimal fix:** replace `→` with ASCII `->` in those four `print()` literals (the `→` characters inside prompt/docstring strings are harmless because they are never written to stdout). Alternatively, force UTF-8 stdout once at process start (`sys.stdout.reconfigure(encoding="utf-8")`), but the ASCII substitution is smaller and platform-proof.
- **Verification that the surrounding logic is fine:** with `PYTHONIOENCODING=utf-8`, the full fallback table passed (see §10 results), and the research nodes passed.

### S3 — MEDIUM: a chart type supplied in a clarification answer is silently discarded

- **Files/lines:** `agents/visualization_agent.py:66-75` (structured-data path) and `_chart_type_from_task` at `:325-340`; relies on `services/message_utils.latest_user_request`.
- **Root cause / cross-file meaning mismatch:** after the clarification fix, `latest_user_request()` intentionally skips tagged clarification answers so the *original* request is used for intent. But `Visualization` then derives chart type **only** from `latest_user_request(state.messages) or state.current_task`. If the original request did not name a type and the user supplied one in the clarification answer ("…, pie chart"), that answer is never consulted, so `_chart_type_from_task` falls back to `"bar"`. The title path has the same input but titles are less sensitive.
- **Real reproduction** (end-to-end through `Visualization`, renderer stubbed so no image is written):

```
A: original request did NOT specify type; clarification answer says 'pie'
  chart_type chosen: bar
  chart title: 'chart of our sales by region'
B: control - original request explicitly says 'pie'
  chart_type chosen: pie
C: what latest_user_request resolves to in both cases
  A resolved request: 'Show me a chart of our sales by region'
  B resolved request: 'Show me a pie chart of our sales by region'
```

- **Proposed minimal fix:** keep the original request as the *primary* intent source, but also consider the clarification answers when choosing a chart type, e.g. build a small resolved string for type detection:
  `type_text = " ".join(filter(None, [latest_user_request(state.messages), *clarification_answers(state.messages), state.current_task]))`
  and pass `type_text` to `_chart_type_from_task`. Title stays `_title_from_user_request`. Do **not** revert `latest_user_request` to "last HumanMessage" — that was the original bug.

### S4 — LOW: `run_general_query` validates a *different string* than it executes; over-blocks literal semicolons

- **File/lines:** `tools/database.py:51-62`.
- **Finding A (mismatch):** the guard computes `stripped` (trim + remove one trailing `;`) and validates that, but executes the original `query_text`. Real output:

```
input:     'SELECT 1;   '
validated: 'SELECT 1'   (strip + trailing ';' removed)
executed:  SQL('SELECT 1;   ')
```

  I fuzzed trailing semicolons, whitespace padding, embedded comments and newline-separated statements and could **not** construct a multi-statement bypass — any remaining `;` raises. So this is an inconsistency, not a demonstrated injection. It should still execute the validated string.
- **Finding B (over-block):** the `;` check is not comment/string aware, so legitimate single statements are rejected:

```
EXECUTED trailing semicolon        'SELECT 1;'                       -> [{'?column?': 1}]
REJECTED two statements            'SELECT 1; SELECT 2'              -> ValueError
REJECTED newline second statement  'SELECT 1;\nDROP TABLE employees' -> ValueError
REJECTED semicolon in string lit   "SELECT ';'"                      -> ValueError
EXECUTED semicolon in line comment 'SELECT 1 -- ;'                   -> [{'?column?': 1}]
```

- **Proposed minimal fix:** execute the validated copy (`cur.execute(sql.SQL(stripped))`). For the false positive, either document the limitation or do a lightweight comment/quote-aware scan before the `;` check. Note the real boundary is the Postgres role, so this is correctness/UX, not a security hole.

### S5 — LOW: `bool` is accepted as a numeric chart value

- **File/lines:** `agents/sql_agent.py:10-25` (`isinstance(value, (int, float))`).
- **Real output:**

```
bool value       -> [{'label': 'flag', 'value': 1.0}]
bool label/value -> [{'label': 'False', 'value': 1.0}]
```

  A boolean column returned by `run_general_query` is silently charted as `1.0/0.0`. `Decimal` is handled correctly, `None`/1-column/3-column/reversed-order/non-list all correctly return `None`.
- **Proposed minimal fix:** if booleans are not meaningful chart quantities, reject explicitly before the numeric check (`if isinstance(value, bool): return None`), or coerce/document intentionally.

### S6 — INFO / live risk: unregistered Pydantic types silently degrade to `dict` in the checkpointer

- **File:** `services/memory.py:16-20` (`allowed_msgpack_modules=[TaskRecord, SpecialistResult]`).
- **Audit result:** the allowlist is **exactly** the set of custom models ever placed in a checkpointed `SupervisorState` channel: `last_result: SpecialistResult` and `task_history: List[TaskRecord]`. All other models (`ChartSpec`, `FactExtraction`, `SupervisorDecision`, `ClarificationOutput`, `ReportOutput`, `SubDecision`) are transient, and `SubGraphSupervisorState` is compiled without a checkpointer. Allowlist correct, no more/no less.
- **Round-trips through the actual serde** (`dumps_typed`/`loads_typed`) all equal: `SpecialistResult`, `TaskRecord`, `list[TaskRecord]`, nested `{"last_result":…, "task_history":[…]}`; and `PostgresSaver` preserves `TaskRecord` across a real checkpoint.
- **Unregistered behaviour (confirmed):**

```
ChartSpec (unregistered):      wire_type='msgpack' back_type=dict  degraded=True
FactExtraction (unregistered): wire_type='msgpack' back_type=dict  degraded=True
Blocked deserialization of state.state.ChartSpec - not in allowed_msgpack_modules...
```

  Harmless today, but the next person who adds a checkpointed field of a new custom type gets a silent `dict` and a confusing `AttributeError` downstream, not an exception at write time. Consider a serializer test asserting the allowlist covers every channel model.

### S7 — LOW: test-file hygiene gaps

- `tests/test_rag_agent.py:6-32` — two `@pytest.mark.integration` tests call `RAG(state)`, which makes a **real LLM** call, but the file has **no** `requires_llm` / `requires_db` guards (unlike the `test_llm_*` suites). In an environment without credentials/DB they fail instead of skipping. Not run here.
- `tests/test_sql_agent.py:47-55` — `test_general_permission_does_not_block_nonsensitive_keywords` is `@pytest.mark.integration` and invokes `Sql_agent` with a **real LLM**; its name says "does not block" while its assertion is `issue == "permission_denied"`. Worth reviewing the naming/intent while it is still unrun. Not run here.

---

## Sections executed and results

| § | Area | Result |
|---|---|---|
| §3 | Cross-file state consistency sweep | All fields (`clarification_question`, `clarification_count`, `conversation_summary`, `research_*`, `turn_count`, `task_history`, `last_result`, `current_task`, `structured_data`, `route`) have consistent readers/writers. `latest_user_request` is the only "latest request" resolver on the decision path; `finalize._latest_human_message` (memory extraction) and `research.supervisor._clean_latest_content` (sub-controller note) are different, legitimate uses. `TaskRecord.route` still `SpecialistRoute`. |
| §4 | Checkpoint serializer audit | Pass (see S6). |
| §5 | Clarification matrix | Pass, except S3. Zero LLM calls in `Clarification` (node has no `llm` symbol; call delta across resume was exactly 2: one `SupervisorDecision` + one `convo`); `memory_manager` does not re-run on resume (`turn_count` stayed 1); the interrupt payload question is what lands in the transcript; cap routes to a user-visible `convo` reply with no second interrupt; `ClarificationOutput` generated once. |
| §6 | Research subgraph | Pass. Threshold is strict `> 800`; cap halts at `MAX_RESEARCH_ATTEMPTS=3` and still writes a report; substantial-note short-circuit makes only 1 `SubDecision` call; immediate `"end"` → graceful `failed/research_no_results` (no `IndexError`); whitespace-only → `research_empty_report`; `succeeded=False` → failed; boundary returns only `last_result` (no message bleed). |
| §7 | SQL / RAG specialists | Pass, except S5. Duplicate tool loops → `failed/empty_synthesis` **with `structured_data` preserved**; empty result set → `done` text; missing table → `failed/table_or_schema_missing`; RAG `rag_unavailable` vs `no_matching_documents` vs `done` all correct. **Real retrieval**: relevant query distances `[0.439, 0.690, 0.793]` all `< 0.8`; out-of-corpus min distance `0.873 > 0.8`. |
| §8 | Message-list integrity | Pass. Summary is injected before recent messages; `RemoveMessage` prunes the Postgres/Memory checkpoint (12→7 messages on the summarization turn); a second summarization folds the prior summary into the new transcript; `clarification_count` resets each turn; `Finalize` suppresses only exact-content duplicates and adds the answer when the last AI message differs. |
| §9 | Concurrency / SSRF / SQL guard | Pass, except S4. Full SSRF table blocked (loopback v4/v6, link-local/metadata, private, reserved, multicast, unspecified, bad schemes, bad ports, no hostname); public IP allowed. Concurrent `fetch_page` calls with two different pinned IPs serialized under the lock, each saw its own pin, and `urllib3_conn.create_connection` was restored exactly. |
| §10 | LLM fallback/retry | Pass (under UTF-8). `{429,500,502,503,504,529}` fall back; `{400,401,403,422}` raise immediately even with a poison message full of retryable substrings; `404` retryable only with a decommission heuristic marker; no-status `ConnectionError`/`TimeoutError` fall back; no-status `ValueError` raises; all-fail raises one `RuntimeError` naming the last error. Nested `.response.status_code` handled. **Caveat:** S2 makes this path crash on cp1252 before the fallback can occur. |
| §11 | Existing suite | Default: 63 passed / 29 deselected / 1 xfailed / 0 failed (current tree). All 29 deselected are `llm`/`integration`. |

## Files reviewed with no issues found (and what was tried)

- `graph/routing.py` — exercised `None` and corrupted `next` for both routers (raises clearly); valid routes pass.
- `graph/workflow.py` — main graph and research subgraph compile; node/edge wiring traced.
- `state/state.py`, `state/structure_output.py` — every field/type traced to callers; `model_construct` not used anywhere in production.
- `services/message_utils.py` — `content_to_text` for str/list/None; `latest_user_request` skips tagged answers and falls back to the most recent answer only when no true request exists; multimodal content normalized.
- `services/errors.py` — status-first classification; bare digits in unrelated text not misclassified.
- `services/memory.py` — allowlist exact; `write_fact`/`read_facts`/`format_facts_for_prompt` exercised against real Postgres; PostgresSaver tuple API and `.serde` assignment work.
- `agents/supervisor.py` — deterministic guards, hop cap, same-route cap, duplicate-task guard, `map_to_state`; the LLM-skip tests (done→end, auto-visu) pass with `llm` raising if called. (Cosmetic only: unused `ValidationError` import.)
- `agents/clarification.py` — no LLM symbol at all; transcript correct; count increments.
- `agents/conversation_agent.py`, `agents/finalize.py` — message emission and exact-duplicate suppression verified; the redundant branch removed externally is behavior-neutral (and slightly safer for empty content).
- `agents/research/supervisor.py`, `researcher.py`, `report_writer.py`, `research_node.py` — see §6; the only defect there is the encoding issue (S2).
- `tools/web_search.py` — see §9 (SSRF + concurrency); no live network was used.
- `tools/database.py` — role boundary enforced by Postgres (general_role cannot `SELECT salaries/credentials`); see S4 for the validation-string issue.
- `db/connection.py` — `connect_timeout` addition verified; real general/elevated connections succeed.
- `rag/retrieval.py`, `rag/indexing.py` — real embedder + real pgvector retrieval; embeddings upsert works.
- `main.py`, `chat.py` — traced the interrupt/resume loop and config plumbing (not run as an app; no Chainlit server started).
- `agents/visualization_agent.py` — deterministic structured-data path, prefix stripping, pie validation; see S3.

## Unverified / gaps (not silently omitted)

- **All `llm`-marked tests** (`test_llm_rag.py`, `test_llm_research.py`, `test_llm_sql.py`, `test_llm_supervisor_routing.py`) and the two `test_rag_agent.py` integration tests and `test_sql_agent.py::test_general_permission_does_not_block_nonsensitive_keywords` — **not run**, because they invoke the real configured model (no live LLM calls permitted by the task). Their logic was instead driven with the fake harness where applicable.
- **Live outbound HTTP** (`web_search` via DDGS, `fetch_page` real transport, redirect handling, content-type/size handling) — **not exercised**. Only `_resolve_and_validate` and a fake `requests.get` were used; no external requests were made.
- **PostgresSaver under concurrency / multiple workers** — a single-threaded real round-trip passed; concurrent checkpoint writers were not tested.
- **Chainlit app runtime** (`chat.py` auth/data-layer, `services/auth.py`) — read-only; no server was started.
- **`requirements.txt` reinstall path** — file is UTF-16LE (BOM `FF FE`) and would not parse under a plain `pip install -r`; the pre-existing `.venv` already satisfies every pin, so no install/convert was attempted.
- **`test_llm_research.py::test_llm_sub_controler_ends_after_report_written_without_llm`** is structurally LLM-free yet deselected behind the `llm` marker; its behavior is covered by `test_sub_controler.py` and the §6 subgraph run.

## Cleanup note

`docker compose up -d postgres` was started for this review; container `silicon_cedars_db` is left running. The 9 rows in `general_embeddings` were generated by `rag.indexing.index_lessons_learned()` and a `qa-pg-thread-1` checkpoint row exists in the checkpoint tables. No repository files were changed by this pass.
