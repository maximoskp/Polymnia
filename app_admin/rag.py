
import os
import requests

from pathlib import Path

import chromadb


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DB_DIR = (
    BASE_DIR
    / "data"
    / "odigos_dioikitikon_diadikasion"
    / "vectordb"
    / "chroma"
)
print("ChromaDB location:", DB_DIR.resolve())

print("Database directory:", DB_DIR.resolve())
print("Exists:", DB_DIR.exists())
print("SQLite exists:", (DB_DIR / "chroma.sqlite3").exists())

print("\nDirectory contents:")

for item in DB_DIR.iterdir():
    print(
        item.name,
        "[DIR]" if item.is_dir() else "[FILE]"
    )

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

print("\nCollections:")

for collection in client.list_collections():
    print(collection.name)

print("\nConnection successful.")

COLLECTION_NAME = "hmu_admin_documents"

# Ollama configuration

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

EMBEDDING_MODEL = "bge-m3"

KEEP_ALIVE = "30m"

REQUEST_TIMEOUT = 120

EMBEDDING_DIM = 1024

# --------------------------------------------------
# Initialization
# --------------------------------------------------

collection = None


def initialize_retriever():

    global collection

    if collection is not None:
        return

    print("Connecting to Ollama...")

    response = requests.get(
        f"{OLLAMA_HOST}/api/tags",
        timeout=10
    )

    response.raise_for_status()

    available_models = [
        item["name"].split(":")[0]
        for item in response.json().get("models", [])
    ]

    if EMBEDDING_MODEL not in available_models:

        raise RuntimeError(
            f"Embedding model {EMBEDDING_MODEL} "
            "is not installed in Ollama."
        )

    print("Ollama connection established.")

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Retriever ready. "
        f"Documents/chunks: {collection.count()}"
    )

# --------------------------------------------------
# Embedding generation
# --------------------------------------------------

def embed_text(text):

    response = requests.post(

        f"{OLLAMA_HOST}/api/embed",

        json={
            "model": EMBEDDING_MODEL,
            "input": text,
            "truncate": False,
            "keep_alive": KEEP_ALIVE
        },

        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    if len(embeddings) != 1:

        raise RuntimeError(
            "Expected exactly one embedding."
        )

    embedding = embeddings[0]

    if len(embedding) != EMBEDDING_DIM:

        raise RuntimeError(
            f"Unexpected embedding dimension: "
            f"{len(embedding)}"
        )

    return embedding

# --------------------------------------------------
# Semantic retrieval
# --------------------------------------------------

def search(query, n_results=5):

    if collection is None:
        raise RuntimeError("Retriever not initialized")

    count = collection.count()

    if count == 0:
        return []

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, count),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        distance = results["distances"][0][i]

        document_id = metadata.get("document_id", "")

        # Local article URL when the document ID follows
        # our existing hmu_odigos_arthro_XX convention.
        article_url = None

        if document_id.startswith("hmu_odigos_arthro_"):
            article_id = document_id.rsplit("_", 1)[-1]

            if article_id.isdigit() and 1 <= int(article_id) <= 34:
                article_url = f"/articles/{int(article_id):02d}"

        retrieved.append({

            "text": document,

            "document_title": metadata.get(
                "document_title", "Άγνωστο έγγραφο"
            ),

            "section": metadata.get(
                "section", ""
            ),

            "source_url": metadata.get(
                "source_url", ""
            ),

            "document_id": document_id,

            "article_url": article_url,

            "line_start": metadata.get("line_start"),

            "line_end": metadata.get("line_end"),

            "similarity": round(1.0 - distance, 4)

        })

    return retrieved
