from __future__ import annotations

import pytest

from db.connection import _get_general_pool, get_general_connection
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]


def test_general_pool_reuses_bounded_connections():
    """
    Sequential acquisitions should be served from the bounded pool rather
    than growing past pool_max. Skipped automatically when no DB is
    reachable (see requires_db), so CI without Postgres is unaffected.
    """
    pool = _get_general_pool()

    for _ in range(2):
        with get_general_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

    stats = pool.get_stats()
    assert stats["pool_size"] >= 1
    assert stats["pool_size"] <= stats["pool_max"]
