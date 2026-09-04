import os
import psycopg
from dotenv import load_dotenv

load_dotenv()


def get_general_connection():
    """
    Connection for the free-form SQL agent path. Uses general_role,
    which has SELECT only on non-sensitive tables. Sensitive tables
    are not queryable through this connection at all.
    """
    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "company_intel"),
        user=os.getenv("DB_GENERAL_USER", "general_role"),
        password=os.getenv("DB_GENERAL_PASSWORD"),
    )


def get_elevated_connection():
    """
    Connection for the fixed, gated tools only (get_salary,
    get_user_credential). Never used for free-form/agent-generated SQL.
    """
    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "company_intel"),
        user=os.getenv("DB_ELEVATED_USER", "app_owner"),
        password=os.getenv("DB_ELEVATED_PASSWORD"),
    )