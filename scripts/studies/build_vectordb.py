
import json
import torch
import chromadb
import requests

from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = BASE_DIR / "data/studies/chunks/segments.jsonl"
DB_DIR = BASE_DIR / "data/studies/vectordb/chroma"

COLLECTION_NAME = "hmu_student_documents"

OLLAMA_HOST = "http://localhost:11434"

EMBEDDING_MODEL = "bge-m3"

KEEP_ALIVE = "30m"

REQUEST_TIMEOUT = 300

BATCH_SIZE = 4
EMBEDDING_DIM = 1024


# --------------------------------------------------
# Ollama connection and embedding generation
# --------------------------------------------------

def check_ollama():

    print("Connecting to Ollama...")

    response = requests.get(
        f"{OLLAMA_HOST}/api/tags",
        timeout=10
    )

    response.raise_for_status()

    models = response.json().get("models", [])

    available_models = [
        model["name"].split(":")[0]
        for model in models
    ]

    if EMBEDDING_MODEL not in available_models:

        raise RuntimeError(
            f"Model {EMBEDDING_MODEL} not found in Ollama. "
            f"Run: ollama pull {EMBEDDING_MODEL}"
        )

    print(
        f"Ollama ready. "
        f"Embedding model: {EMBEDDING_MODEL}"
    )


def embed_texts(texts):

    response = requests.post(

        f"{OLLAMA_HOST}/api/embed",

        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
            "truncate": False,
            "keep_alive": KEEP_ALIVE
        },

        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    if len(embeddings) != len(texts):

        raise RuntimeError(
            "Number of embeddings does not match "
            "number of input texts."
        )

    for embedding in embeddings:

        if len(embedding) != EMBEDDING_DIM:

            raise RuntimeError(
                f"Unexpected embedding dimension: "
                f"{len(embedding)}"
            )

    return embeddings

# --------------------------------------------------
# Initialize ChromaDB
# --------------------------------------------------

DB_DIR.mkdir(parents=True, exist_ok=True)

print("BUILD DB_DIR:", DB_DIR.resolve())

client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine",
        "description": "HMU administrative documents"
    }
)


# --------------------------------------------------
# Load JSONL segments
# --------------------------------------------------

def load_segments(filepath):

    segments = []

    with open(filepath, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {e}"
                )

            if not item.get("text", "").strip():
                print(f"Skipping empty segment at line {line_number}")
                continue

            # Each segment must have a unique identifier.
            segment_id = item.get("segment_id")

            if not segment_id:
                raise ValueError(
                    f"Missing segment_id at line {line_number}"
                )

            item["_vector_id"] = segment_id

            segments.append(item)

    # Detect duplicate segment IDs.
    ids = [item["_vector_id"] for item in segments]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate segment IDs found in JSONL.")

    return segments


# --------------------------------------------------
# Prepare metadata
# --------------------------------------------------

def prepare_metadata(item):

    metadata = {

        "document_id": item.get("document_id"),

        "document_title": item.get("document_title"),

        "source_url": item.get("source_url"),

        "article_number": item.get("article_number"),

        "section": item.get("section"),

        "chunk_id": item.get("chunk_id"),

        "parent_chunk_id": item.get("parent_chunk_id"),

        "segment_index": item.get("segment_index"),

        "segment_count": item.get("segment_count"),

        "chunk_index": item.get("chunk_index"),

        "line_start": item.get("line_start"),

        "line_end": item.get("line_end"),

        "token_count": item.get("token_count"),

        "source_file": item.get("source_file"),

        "page_start": item.get("page_start"),

        "page_end": item.get("page_end"),

        # Store complex structures as JSON strings.
        "section_path": json.dumps(
            item.get("section_path", []),
            ensure_ascii=False
        ),

        "source_links": json.dumps(
            item.get("source_links", []),
            ensure_ascii=False
        )
    }

    # ChromaDB metadata must not contain None values.
    return {
        key: value
        for key, value in metadata.items()
        if value is not None
    }


# --------------------------------------------------
# Generate embeddings and insert into ChromaDB
# --------------------------------------------------

def build_database():

    segments = load_segments(INPUT_FILE)

    print(f"Loaded {len(segments)} segments.")

    for start in tqdm(
        range(0, len(segments), BATCH_SIZE),
        desc="Embedding and indexing"
    ):

        batch = segments[start:start + BATCH_SIZE]

        texts = [
            item["text"]
            for item in batch
        ]

        # Generate dense embeddings through Ollama.
        embeddings = embed_texts(texts)

        ids = [
            item["_vector_id"]
            for item in batch
        ]

        metadatas = [
            prepare_metadata(item)
            for item in batch
        ]

        # Upsert makes the operation repeatable.
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print("\nDatabase construction completed.")

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Segments in source: {len(segments)}")
    print(f"Vectors in database: {collection.count()}")
    print(f"Database location: {DB_DIR}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    check_ollama()

    build_database()
