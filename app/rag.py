
import os

# Offline model loading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import chromadb
from FlagEmbedding import BGEM3FlagModel


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

COLLECTION_NAME = "hmu_admin_documents"

MODEL_PATH = (
    "/home/maximos/.cache/huggingface/hub/"
    "models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)

MAX_LENGTH = 8192


# --------------------------------------------------
# Initialization
# --------------------------------------------------

model = None
collection = None


def initialize_retriever():

    global model, collection

    if model is not None:
        return

    print("Loading BGE-M3...")

    model = BGEM3FlagModel(
        MODEL_PATH,
        use_fp16=True,
        devices="cuda:0"
    )

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
# Semantic retrieval
# --------------------------------------------------

def search(query, n_results=5):

    if model is None or collection is None:
        raise RuntimeError("Retriever not initialized")

    count = collection.count()

    if count == 0:
        return []

    result = model.encode(
        [query],
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )

    query_embedding = result["dense_vecs"].tolist()

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
