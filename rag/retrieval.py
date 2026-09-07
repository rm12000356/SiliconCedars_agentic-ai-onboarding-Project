from pgvector.psycopg import register_vector
from db.connection import get_general_connection
from rag.indexing import get_embedder


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list[dict]:
    """
    Embeds the query and finds the top_k closest chunks in
    general_embeddings by cosine distance. Uses the general_role
    connection, this is a read-only, agent-facing operation, same
    tier as run_general_query.
    """
    embedder = get_embedder()
    query_vector = embedder.embed_query(query)

    with get_general_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, source_table, source_id, embedding <=> %s::vector AS distance
                FROM general_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, top_k),
            )
            rows = cur.fetchall()

    return [
        {"content": r[0], "source_table": r[1], "source_id": r[2], "distance": r[3]}
        for r in rows
    ]