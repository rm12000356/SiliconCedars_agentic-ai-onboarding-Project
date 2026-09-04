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