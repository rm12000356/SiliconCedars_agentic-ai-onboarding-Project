from langchain_huggingface import HuggingFaceEmbeddings
from pgvector.psycopg import register_vector
from db.connection import get_elevated_connection

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None


def get_embedder():
    """
    Cached embedder instance, loading the model is slow, don't reload
    it on every call.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def index_lessons_learned():
    """
    Pulls all rows from lessons_learned, embeds lesson_text, and
    upserts into general_embeddings. Run manually as a standalone
    script whenever the source data changes, not invoked by the graph.

    Uses the elevated connection deliberately: writing embeddings is
    an admin/ingestion operation, not something the agent-facing
    general_role should be able to do. general_role is read-only by
    design, same principle as the SQL agent's connection split.
    """
    embedder = get_embedder()

    with get_elevated_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id, lesson_text FROM lessons_learned")
            rows = cur.fetchall()

            for row_id, lesson_text in rows:
                vector = embedder.embed_query(lesson_text)

                cur.execute(
                    "SELECT id FROM general_embeddings WHERE source_table = %s AND source_id = %s",
                    ("lessons_learned", row_id),
                )
                existing = cur.fetchone()

                if existing:
                    cur.execute(
                        "UPDATE general_embeddings SET content = %s, embedding = %s WHERE id = %s",
                        (lesson_text, vector, existing[0]),
                    )
                else:
                    cur.execute(
                        "INSERT INTO general_embeddings (source_table, source_id, content, embedding) "
                        "VALUES (%s, %s, %s, %s)",
                        ("lessons_learned", row_id, lesson_text, vector),
                    )

        conn.commit()

    print(f"Indexed {len(rows)} lessons_learned rows into general_embeddings.")


if __name__ == "__main__":
    index_lessons_learned()