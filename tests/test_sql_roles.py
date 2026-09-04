from db.connection import get_general_connection


def test_general_role_cannot_query_salaries():
    with get_general_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT * FROM salaries")
                assert False, "general_role should NOT be able to query salaries"
            except Exception as e:
                print(f"Expected failure: {e}")


def test_general_role_cannot_see_salaries_in_metadata():
    with get_general_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name = 'salaries'"
            )
            row = cur.fetchone()
            assert row is None, "salaries table should be invisible to general_role, but it was found"
            print("Confirmed: salaries is invisible to general_role")


if __name__ == "__main__":
    test_general_role_cannot_query_salaries()
    test_general_role_cannot_see_salaries_in_metadata()
    print("All role-boundary tests passed.")