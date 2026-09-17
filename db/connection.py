import atexit
import os
import threading

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


_POOL_MIN_SIZE = _int_env("DB_POOL_MIN_SIZE", 1)
_POOL_MAX_SIZE_GENERAL = _int_env("DB_POOL_MAX_SIZE_GENERAL", 5)
_POOL_MAX_SIZE_ELEVATED = _int_env("DB_POOL_MAX_SIZE_ELEVATED", 3)
_POOL_ACQUIRE_TIMEOUT = _float_env("DB_POOL_ACQUIRE_TIMEOUT", 30.0)
_CONNECT_TIMEOUT = _int_env("DB_CONNECT_TIMEOUT", 5)

_POOLS_LOCK = threading.Lock()
_general_pool: ConnectionPool | None = None
_elevated_pool: ConnectionPool | None = None


def _base_kwargs() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "company_intel"),
        "connect_timeout": _CONNECT_TIMEOUT,
    }


def _get_general_pool() -> ConnectionPool:
    global _general_pool
    if _general_pool is None:
        with _POOLS_LOCK:
            if _general_pool is None:
                kwargs = _base_kwargs()
                kwargs.update(
                    user=os.getenv("DB_GENERAL_USER", "general_role"),
                    password=os.getenv("DB_GENERAL_PASSWORD"),
                    options="-c statement_timeout=5000",
                )
                _general_pool = ConnectionPool(
                    kwargs=kwargs,
                    min_size=_POOL_MIN_SIZE,
                    max_size=_POOL_MAX_SIZE_GENERAL,
                    timeout=_POOL_ACQUIRE_TIMEOUT,
                    open=True,
                    name="general",
                )
    return _general_pool


def _get_elevated_pool() -> ConnectionPool:
    global _elevated_pool
    if _elevated_pool is None:
        with _POOLS_LOCK:
            if _elevated_pool is None:
                kwargs = _base_kwargs()
                kwargs.update(
                    user=os.getenv("DB_ELEVATED_USER", "app_owner"),
                    password=os.getenv("DB_ELEVATED_PASSWORD"),
                )
                _elevated_pool = ConnectionPool(
                    kwargs=kwargs,
                    min_size=_POOL_MIN_SIZE,
                    max_size=_POOL_MAX_SIZE_ELEVATED,
                    timeout=_POOL_ACQUIRE_TIMEOUT,
                    open=True,
                    name="elevated",
                )
    return _elevated_pool


def get_general_connection(acquire_timeout: int | None = None):
    """General-role connection for the free-form SQL path (no access to
    sensitive tables). `acquire_timeout` bounds pool wait only; the libpq
    connect attempt is bounded separately by DB_CONNECT_TIMEOUT."""
    pool = _get_general_pool()
    if acquire_timeout is None:
        return pool.connection()
    return pool.connection(timeout=acquire_timeout)


def get_elevated_connection():
    """Elevated connection for the fixed, gated tools only (get_salary,
    get_user_credential); never used for free-form SQL."""
    return _get_elevated_pool().connection()


def close_pools() -> None:
    """Close both pools and drop the cached handles (debug/shutdown helper)."""
    global _general_pool, _elevated_pool
    with _POOLS_LOCK:
        for pool in (_general_pool, _elevated_pool):
            if pool is not None:
                pool.close()
        _general_pool = None
        _elevated_pool = None


atexit.register(close_pools)
