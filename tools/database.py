from psycopg import sql
from db.connection import get_elevated_connection, get_general_connection
from langchain.tools import tool

MAX_ROWS = 5000
MAX_RESULT_BYTES = 1_000_000

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
    """Check whether a credential record exists for user_id. Returns {"found": bool}."""
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
    if not query_text or not query_text.strip():
        raise RuntimeError(
            "run_general_query called with an empty query_text. A valid "
            "SQL statement is required."
        )
    
    stripped = query_text.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1]
    # Not quote/comment aware, so a literal ';' inside a string or comment is
    # rejected too. The real boundary is the Postgres role; this is a
    # convenience guard, not a security control.
    if ";" in stripped:
        raise ValueError(
            "Multiple SQL statements are not allowed in a single query. "
            "Run one statement at a time."
        )

    with get_general_connection() as conn:
        with conn.cursor() as cur:
            # Read-only transaction: blocks writes, DDL, and temp tables at
            # the session level (the role grants alone do not).
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(sql.SQL(stripped))

            if cur.description is None:
                return []

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchmany(MAX_ROWS + 1)
            if len(rows) > MAX_ROWS:
                raise ValueError(
                    f"Query returned more than {MAX_ROWS} rows. "
                    "Add a LIMIT or narrow the request."
                )

            result: list[dict] = []
            total_bytes = 0
            for row in rows:
                item = dict(zip(columns, row))
                total_bytes += len(str(item))
                if total_bytes > MAX_RESULT_BYTES:
                    raise ValueError(
                        "Query result is too large to return. "
                        "Narrow the request or select fewer columns."
                    )
                result.append(item)
            return result

GENERAL_TOOLS = [run_general_query]
ELEVATED_TOOLS = [get_salary, get_user_credential]