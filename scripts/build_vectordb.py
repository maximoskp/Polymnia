
import json
import torch
import chromadb

from pathlib import Path
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data/odigos_dioikitikon_diadikasion/chunks/segments.jsonl"
DB_DIR = BASE_DIR / "data/odigos_dioikitikon_diadikasion/vectordb/chroma"

COLLECTION_NAME = "hmu_admin_documents"

MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE = 4
MAX_LENGTH = 8192


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

device_is_gpu = torch.cuda.is_available()

print(f"Loading embedding model: {MODEL_NAME}")

model = BGEM3FlagModel(
    MODEL_NAME,
    use_fp16=device_is_gpu
)


# --------------------------------------------------
# Initialize ChromaDB
# --------------------------------------------------

DB_DIR.mkdir(parents=True, exist_ok=True)

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

        # Generate dense embeddings.
        result = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )

        embeddings = result["dense_vecs"].tolist()

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
    try:
        build_database()

    finally:
        try:
            if getattr(model, "pool", None) is not None:
                model.stop_self_pool()
        except Exception as e:
            print(f"Model cleanup warning: {e}")

        del model
