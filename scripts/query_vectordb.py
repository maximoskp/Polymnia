
import torch
import chromadb

from pathlib import Path
from FlagEmbedding import BGEM3FlagModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DB_DIR = BASE_DIR / "data/odigos_dioikitikon_diadikasion/vectordb/chroma"

COLLECTION_NAME = "hmu_admin_documents"

MODEL_NAME = "BAAI/bge-m3"

MAX_LENGTH = 8192


# --------------------------------------------------
# Load model and database
# --------------------------------------------------

model = BGEM3FlagModel(
    MODEL_NAME,
    use_fp16=torch.cuda.is_available()
)

client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# --------------------------------------------------
# Semantic search
# --------------------------------------------------

def search(query, n_results=5, document_id=None):

    result = model.encode(
        [query],
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )

    query_embedding = result["dense_vecs"].tolist()

    search_args = {
        "query_embeddings": query_embedding,
        "n_results": min(n_results, collection.count()),
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    # Optional document-level filtering.
    if document_id:
        search_args["where"] = {
            "document_id": document_id
        }

    results = collection.query(**search_args)

    for i in range(len(results["ids"][0])):

        metadata = results["metadatas"][0][i]

        distance = results["distances"][0][i]

        similarity = 1.0 - distance

        print("\n" + "=" * 70)

        print(f"Result: {i + 1}")
        print(f"Similarity: {similarity:.4f}")

        print(
            f"Document: {metadata.get('document_title')}"
        )

        print(
            f"Section: {metadata.get('section')}"
        )

        print(
            f"Source: {metadata.get('source_url')}"
        )

        print(
            f"Lines: {metadata.get('line_start')} - "
            f"{metadata.get('line_end')}"
        )

        print("\nText:")
        print(results["documents"][0][i])


# --------------------------------------------------
# Main
# --------------------------------------------------


if __name__ == "__main__":

    try:
        while True:
            question = input("\nEnter your question (or 'exit'): ")

            if question.strip().lower() in {"exit", "quit"}:
                break

            if question.strip():
                search(
                    query=question,
                    n_results=5
                )

    finally:
        print("\nClosing embedding model...")

        try:
            model.stop_self_pool()
        except Exception as e:
            print(f"Cleanup warning: {e}")

        print("Shutdown completed.")


'''
search(
    query="What are the deadlines for faculty recruitment?",
    n_results=5,
    document_id="hmu_odigos_arthro_01"
)
'''