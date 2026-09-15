-- ============================================================
-- App users (auth source of truth for Chainlit + memory)
-- ============================================================
-- user_id used by long-term memory (user_memory) and LangGraph
-- config is app_users.username (TEXT). Keep that convention
-- everywhere so short-term threads and long-term facts stay aligned.
-- ============================================================

CREATE TABLE IF NOT EXISTS app_users (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username         TEXT NOT NULL UNIQUE,
    password_hash    TEXT NOT NULL,
    permission_level TEXT NOT NULL DEFAULT 'general'
                     CHECK (permission_level IN ('general', 'elevated')),
    role             TEXT NOT NULL DEFAULT 'user',
    display_name     TEXT,
    is_active        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_app_users_username
    ON app_users (username);

CREATE INDEX IF NOT EXISTS idx_app_users_active
    ON app_users (is_active)
    WHERE is_active = TRUE;

-- ------------------------------------------------------------
-- Seed demo users
-- ------------------------------------------------------------
-- Passwords (plain, for local demo only):
--   alice  / alice123  → elevated / manager
--   bob    / bob123    → general  / analyst
--   admin  / admin     → elevated / admin
--
-- password_hash values below are bcrypt hashes of those passwords.
-- If login fails after first run, regenerate hashes with:
--
--   python -c "import bcrypt; print(bcrypt.hashpw(b'alice123', bcrypt.gensalt()).decode())"
--   python -c "import bcrypt; print(bcrypt.hashpw(b'bob123', bcrypt.gensalt()).decode())"
--   python -c "import bcrypt; print(bcrypt.hashpw(b'admin', bcrypt.gensalt()).decode())"
--
-- Then UPDATE app_users SET password_hash = '...' WHERE username = '...';
-- ------------------------------------------------------------

INSERT INTO app_users (username, password_hash, permission_level, role, display_name)
VALUES
    (
        'alice',
        '$2b$12$81FA.YQvIDemYyAhAN6OIe3FQrIGdl0/MlFEy1peFWzh4C43n8KLK',
        'elevated',
        'manager',
        'Alice'
    ),
    (
        'bob',
        '$2b$12$zcrpiEs7kCBDYskxqcvhw.2WSMVKmQnKDSuJpxRvhmsUYSi4jUlF.',
        'general',
        'analyst',
        'Bob'
    ),
    (
        'admin',
        '$2b$12$eUfOzLUPmJugePb6GrdchuB7V7mlEkWwwmkQ.uVXmDJE5ndudhRme',
        'elevated',
        'admin',
        'Admin'
    )
ON CONFLICT (username) DO NOTHING;

