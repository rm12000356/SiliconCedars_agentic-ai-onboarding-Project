"""DB role boundary. Requires a live Postgres with 02_roles.sql applied."""

import pytest
from db.connection import get_general_connection
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]


def test_general_role_cannot_query_salaries():
    with get_general_connection() as conn:
        with conn.cursor() as cur:
            with pytest.raises(Exception):
                cur.execute("SELECT * FROM salaries")
                cur.fetchall()


@pytest.mark.parametrize("table_name", ["salaries", "credentials"])
def test_general_role_cannot_see_sensitive_tables_in_metadata(table_name):
    with get_general_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = %s",
                (table_name,),
            )
            row = cur.fetchone()
            assert row is None, (
                f"{table_name} should be invisible to general_role, but it was found"
            )


def test_general_query_rejects_ddl_and_temp_tables():
    from tools.database import run_general_query

    with pytest.raises(Exception):
        run_general_query.invoke(
            {"query_text": "CREATE TEMP TABLE p0_temp_probe (x int)"}
        )


def test_general_query_rejects_oversized_result():
    from tools.database import run_general_query

    with pytest.raises(Exception):
        run_general_query.invoke(
            {"query_text": "SELECT generate_series(1, 100000)"}
        )