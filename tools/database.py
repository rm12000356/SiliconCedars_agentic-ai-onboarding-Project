from psycopg import sql
from db.connection import get_elevated_connection, get_general_connection

# Static data for the lessons_learned table. This is a fixed, gated tool, so it uses the elevated connection.

def get_salary(employee_id: int) -> dict:
    """
    Fixed, gated tool. Uses the elevated connection since salary data
    is sensitive. This function does NOT check whether the caller is
    authorized to see this employee's salary, that's the application/
    backend's job, enforced before this tool is ever called. This
    function only handles the actual data access.
    """
    with get_elevated_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT salary FROM salaries WHERE employee_id = %s",
                (employee_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {"found": False, "salary": None}
            return {"found": True, "salary": row[0]}


def get_user_credential(user_id: int) -> dict:
    """
    Same pattern as get_salary. Placeholder for whatever real
    credential lookup this eventually needs to do.
    """
    with get_elevated_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash FROM credentials WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {"found": False}
            return {"found": True, "password_hash": row[0]}


def run_general_query(query_text: str) -> list[dict]:
    """
    Executes agent-generated SQL through the restricted general_role
    connection. Security here comes entirely from the role's grants,
    not from parsing or validating the SQL text itself.
    """
    if sql is None:
            raise RuntimeError(
                "run_general_query called with sql=None. A valid SQL statement is required."
            )
    with get_general_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL(query_text))

            if cur.description is None:
                # No results to fetch (e.g., for INSERT, UPDATE, DELETE)
                return []

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
