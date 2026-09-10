-- General, non-sensitive tables
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL
);

CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    amount NUMERIC NOT NULL,
    region TEXT NOT NULL,
    sale_date DATE NOT NULL
);

-- Sensitive tables, never exposed to the restricted role
CREATE TABLE salaries (
    employee_id INTEGER REFERENCES employees(id),
    salary NUMERIC NOT NULL
);

CREATE TABLE credentials (
    user_id INTEGER REFERENCES employees(id),
    password_hash TEXT NOT NULL
);

CREATE TABLE lessons_learned (
    id SERIAL PRIMARY KEY,
    project_name TEXT NOT NULL,
    lesson_text TEXT NOT NULL,
    tags TEXT[],           -- e.g. {'onboarding', 'architecture', 'process'}
    created_at DATE NOT NULL DEFAULT CURRENT_DATE
);

INSERT INTO lessons_learned (project_name, lesson_text, tags) VALUES
    (
        'Internal Onboarding Bot v1',
        'Splitting SQL access by data sensitivity at the Postgres role level was far more '
        'robust than trying to validate generated SQL in application code. Validation logic '
        'has to anticipate every possible query shape; role-level restriction makes entire '
        'categories of mistake structurally impossible.',
        ARRAY['architecture', 'security']
    ),
    (
        'Customer Support Assistant',
        'State design failed early on because everything was crammed into one shared graph '
        'state. Debugging became difficult since nodes were reading and writing fields that '
        'had nothing to do with their actual responsibility. Splitting shared vs local state '
        'from the start would have avoided this.',
        ARRAY['langgraph', 'state-design']
    ),
    (
        'Internal Onboarding Bot v1',
        'Prompt-only loop prevention proved unreliable. Telling the model "never route to the '
        'same specialist twice" or "stop after a failure" worked most of the time but not '
        'reliably enough, the model would still occasionally loop under slightly different '
        'phrasing of the same failure. The fix was moving loop prevention into deterministic '
        'code guards (attempt counters, task history checks) that run before the model is '
        'even consulted, rather than trusting the model to follow a written rule every time.',
        ARRAY['langgraph', 'reliability', 'supervisor-design']
    ),
    (
        'Internal Onboarding Bot v1',
        'A RAG agent should never fall back to general knowledge when retrieval finds nothing. '
        'Early versions occasionally answered organizational questions from the model''s own '
        'training knowledge when no matching document was found, which produces confident, '
        'plausible-sounding, and potentially wrong answers about internal company specifics. '
        'The fix was reporting retrieval failure as a distinct, structural status rather than '
        'letting the generation step decide how to handle an empty result set.',
        ARRAY['rag', 'grounding', 'reliability']
    ),
    (
        'Internal Onboarding Bot v1',
        'Routing decisions should never be inferred from the tone or wording of a result summary. '
        'A specialist result that included honest caveats or hedging language was sometimes '
        'misread by the orchestrator as an incomplete task needing more work, even when the '
        'result was structurally marked as successful. The fix was adding an explicit rule to '
        'trust the structured status field over the prose content of the summary.',
        ARRAY['architecture', 'state-design']
    ),
    (
        'Internal Onboarding Bot v1',
        'When a subgraph is wrapped as a single node in a parent graph, state must be translated '
        'explicitly at the boundary rather than passed through implicitly by matching field names. '
        'Two state schemas can have fields that share a name but mean different things, and '
        'automatic passthrough can silently feed one schema''s value into a field it was never '
        'meant for, causing validation errors that are confusing to debug.',
        ARRAY['langgraph', 'subgraphs', 'state-design']
    ),
    (
        'Internal Onboarding Bot v1',
        'Credential material, such as password hashes, must never be returned to a model, even '
        'through an internal tool the model calls itself. The correct pattern is an existence or '
        'boolean check only, e.g. "does this credential exist" rather than "here is the value". '
        'Any tool exposing raw secret material to an LLM is a real exposure risk regardless of '
        'how access to that tool is otherwise gated.',
        ARRAY['security', 'credentials']
    ),
    (
        'Internal Onboarding Bot v1',
        'Authorization must never be enforced by the language model''s own judgment, including '
        'keyword-based heuristics. A keyword check for sensitive terms is a reasonable fast-path '
        'convenience, but the actual security boundary has to be enforced deterministically, '
        'for example by which tools are made available to the model at all, or by database-level '
        'role permissions, not by asking the model to decide whether a request is allowed.',
        ARRAY['security', 'authorization']
    ),
    (
        'Internal Onboarding Bot v1',
        'A durable turn counter should never be derived from counting messages in conversation '
        'history. Once old messages are pruned or summarized for context management, a '
        'message-count-based turn number silently changes, desynchronizing it from any '
        'historical record that captured the turn number before pruning happened. Turn '
        'tracking needs its own independent, persisted counter.',
        ARRAY['memory-management', 'state-design']
    );

-- data so queries return something real

INSERT INTO employees (name, department) VALUES
    ('Rami Noueihed', 'Engineering'),
    ('Alice Example', 'Sales');

INSERT INTO sales (amount, region, sale_date) VALUES
    (1200.50, 'MENA', '2026-06-01'),
    (800.00, 'EU', '2026-07-15');

INSERT INTO salaries (employee_id, salary) VALUES
    (1, 95000),
    (2, 72000);

INSERT INTO credentials (user_id, password_hash) VALUES
    (1, 'not_a_real_hash'),
    (2, 'not_a_real_hash_either');