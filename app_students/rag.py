import os
from pathlib import Path

import chromadb
import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Override this with STUDENTS_DB_DIR if the database lives elsewhere.
DB_DIR = Path(
    os.getenv(
        "STUDENTS_DB_DIR",
        str(PROJECT_ROOT / "data" / "studies" / "vectordb" / "chroma"),
    )
)

COLLECTION_NAME = os.getenv(
    "STUDENTS_COLLECTION_NAME",
    "hmu_student_documents",
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
KEEP_ALIVE = os.getenv("EMBEDDING_KEEP_ALIVE", "30m")
REQUEST_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "120"))

collection = None


# --------------------------------------------------
# Initialization
# --------------------------------------------------

def initialize_retriever():
    """Connect to Ollama and the student ChromaDB collection."""
    global collection

    if collection is not None:
        return

    if not DB_DIR.exists():
        raise RuntimeError(
            f"Student ChromaDB directory does not exist: {DB_DIR}. "
            "Set STUDENTS_DB_DIR to the correct location."
        )

    # Check that Ollama is reachable and the embedding model is installed.
    response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
    response.raise_for_status()

    available_models = {
        item["name"]
        for item in response.json().get("models", [])
    }

    model_is_available = any(
        name == EMBEDDING_MODEL or name.split(":")[0] == EMBEDDING_MODEL
        for name in available_models
    )

    if not model_is_available:
        raise RuntimeError(
            f"Embedding model '{EMBEDDING_MODEL}' is not installed in Ollama. "
            f"Available models: {sorted(available_models)}"
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open ChromaDB collection '{COLLECTION_NAME}' "
            f"in {DB_DIR}. Check the collection name and database path."
        ) from exc

    print(
        f"Student retriever ready: {collection.count()} chunks "
        f"in collection '{COLLECTION_NAME}'."
    )


# --------------------------------------------------
# Embedding generation
# --------------------------------------------------

def embed_text(text: str):
    response = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
            "truncate": False,
            "keep_alive": KEEP_ALIVE,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings", [])

    if len(embeddings) != 1:
        raise RuntimeError(
            f"Expected one embedding, received {len(embeddings)}."
        )

    embedding = embeddings[0]

    if len(embedding) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Unexpected embedding dimension: {len(embedding)}; "
            f"expected {EMBEDDING_DIM}."
        )

    return embedding


# --------------------------------------------------
# Semantic retrieval
# --------------------------------------------------

def search(query: str, n_results: int = 5):
    if collection is None:
        raise RuntimeError("Student retriever has not been initialized.")

    count = collection.count()
    if count == 0:
        return []

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}

        retrieved.append({
            "text": document or "",
            "document_title": (
                metadata.get("document_title")
                or metadata.get("title")
                or metadata.get("filename")
                or "Άγνωστο έγγραφο"
            ),
            "section": metadata.get("section") or metadata.get("heading") or "",
            "document_id": metadata.get("document_id", ""),
            "source_url": metadata.get("source_url", ""),
            "similarity": round(1.0 - float(distance), 4),
        })

    return retrieved
