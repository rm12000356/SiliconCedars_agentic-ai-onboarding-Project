from psycopg import sql
from db.connection import get_elevated_connection, get_general_connection
from langchain.tools import tool

# Static data for the lessons_learned table. This is a fixed, gated tool, so it uses the elevated connection.
@tool
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

@tool
def get_user_credential(user_id: int) -> dict:
    """
    """
    with get_elevated_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM credentials WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            return {"found": row is not None}

@tool
def run_general_query(query_text: str) -> list[dict]:
    """
    Executes agent-generated SQL through the restricted general_role
    connection. 
    """
    if not query_text:
        raise RuntimeError(
            "run_general_query called with an empty query_text. A valid "
            "SQL statement is required."
        )
    
    stripped = query_text.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1]
    if ";" in stripped:
        raise ValueError(
            "Multiple SQL statements are not allowed in a single query. "
            "Run one statement at a time."
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

GENERAL_TOOLS = [run_general_query]
ELEVATED_TOOLS = [get_salary, get_user_credential]