import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import chromadb
from FlagEmbedding import BGEM3FlagModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent

DB_DIR = BASE_DIR / "data/odigos_dioikitikon_diadikasion/vectordb/chroma"
COLLECTION_NAME = "hmu_admin_documents"

EMBEDDING_MODEL_PATH = (
    "/home/maximos/.cache/huggingface/hub/"
    "models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)
MAX_EMBEDDING_LENGTH = 8192

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "ilsp/llama-krikri-8b-instruct"
OLLAMA_CONTEXT_LENGTH = 8192
OLLAMA_TEMPERATURE = 0.15
OLLAMA_TIMEOUT = 300

N_RESULTS = 5
MAX_HISTORY_EXCHANGES = 6


# --------------------------------------------------
# Initialize embedding model and vector database
# --------------------------------------------------

print("Loading BGE-M3...")
embedding_model = BGEM3FlagModel(
    EMBEDDING_MODEL_PATH,
    use_fp16=True,
    devices="cuda:0",
)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name=COLLECTION_NAME)

print(f"Connected. Collection contains {collection.count()} chunks.")


# --------------------------------------------------
# Ollama API
# --------------------------------------------------

def ollama_chat(messages, temperature=OLLAMA_TEMPERATURE):
    """Send a non-streaming chat request to the local Ollama API."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": 0,
        "options": {
            "temperature": temperature,
            "num_ctx": OLLAMA_CONTEXT_LENGTH,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Is the Ollama service running?"
        ) from exc

    return result["message"]["content"].strip()


# --------------------------------------------------
# Semantic search
# --------------------------------------------------

def search(query, n_results=N_RESULTS, document_id=None):
    """Retrieve passages and return them as structured records."""

    if collection.count() == 0:
        return []

    result = embedding_model.encode(
        [query],
        max_length=MAX_EMBEDDING_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    query_embedding = result["dense_vecs"].tolist()

    search_args = {
        "query_embeddings": query_embedding,
        "n_results": min(n_results, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }

    if document_id:
        search_args["where"] = {"document_id": document_id}

    results = collection.query(**search_args)

    passages = []

    for i, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i] or {}

        passages.append({
            "rank": i + 1,
            "chunk_id": chunk_id,
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "document_id": metadata.get("document_id"),
            "document_title": metadata.get("document_title"),
            "section": metadata.get("section"),
            "source_url": metadata.get("source_url"),
            "line_start": metadata.get("line_start"),
            "line_end": metadata.get("line_end"),
        })

    return passages


# --------------------------------------------------
# Conversation-aware retrieval
# --------------------------------------------------

def history_as_messages(history):
    """Convert stored exchanges into Ollama chat messages."""

    messages = []

    for exchange in history[-MAX_HISTORY_EXCHANGES:]:
        messages.append({
            "role": "user",
            "content": exchange["question"],
        })
        messages.append({
            "role": "assistant",
            "content": exchange["answer"],
        })

    return messages


def rewrite_question(question, history):
    """
    Resolve references in a follow-up question.
    Return a standalone query suitable for vector retrieval.
    """

    if not history:
        return question

    system_prompt = """
Είσαι ένα σύστημα αναδιατύπωσης ερωτημάτων για το Polymnia,
έναν βοηθό αναζήτησης πληροφοριών σε διοικητικά έγγραφα
του Ελληνικού Μεσογειακού Πανεπιστημίου.

Στόχος σου είναι να μετατρέπεις το τελευταίο ερώτημα του χρήστη
σε ένα αυτοτελές ερώτημα αναζήτησης.

Οδηγίες:
- Χρησιμοποίησε το ιστορικό μόνο για να κατανοήσεις αναφορές
  όπως «αυτό», «εκεί», «και ποια είναι η προθεσμία;».
- Διατήρησε το πραγματικό νόημα και το επίπεδο εξειδίκευσης.
- Μην απαντάς στο ερώτημα.
- Μην προσθέτεις πληροφορίες που δεν υπάρχουν στο ιστορικό.
- Επέστρεψε μόνο το αυτοτελές ερώτημα, κατά προτίμηση στα ελληνικά.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_as_messages(history))
    messages.append({
        "role": "user",
        "content": (
            "Μετέτρεψε το παρακάτω ερώτημα σε αυτοτελές ερώτημα αναζήτησης:\n"
            f"{question}"
        ),
    })

    rewritten = ollama_chat(messages, temperature=0.0)
    return rewritten.strip().strip('"')


# --------------------------------------------------
# RAG answer generation
# --------------------------------------------------

def build_context(passages):
    """Format retrieved chunks and their source metadata for Krikri."""

    blocks = []

    for passage in passages:
        source = passage["document_title"] or "Άγνωστο έγγραφο"
        section = passage["section"] or "Χωρίς ενότητα"

        lines = ""
        if passage["line_start"] is not None:
            lines = (
                f", γραμμές {passage['line_start']}"
                f"-{passage['line_end']}"
            )

        blocks.append(
            f"[{passage['rank']}]\n"
            f"Έγγραφο: {source}\n"
            f"Ενότητα: {section}{lines}\n"
            f"Αναγνωριστικό αποσπάσματος: {passage['chunk_id']}\n"
            f"Περιεχόμενο:\n{passage['text']}"
        )

    return "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(blocks)


def generate_answer(question, passages, history):
    """Generate a concise Greek answer grounded in retrieved passages."""

    if not passages:
        return (
            "Δεν εντοπίστηκαν σχετικά αποσπάσματα στη διαθέσιμη βάση "
            "εγγράφων. Δεν μπορώ να τεκμηριώσω απάντηση με τα διαθέσιμα "
            "στοιχεία."
        )

    system_prompt = """
Είσαι το Polymnia, ένας βοηθός διοικητικής πληροφόρησης
του Ελληνικού Μεσογειακού Πανεπιστημίου.

Απαντάς κυρίως στα ελληνικά, με σαφή, σύντομο και επαγγελματικό τρόπο.

Βασικοί κανόνες:
1. Απάντησε αποκλειστικά με βάση τα αποσπάσματα των εγγράφων
   που παρέχονται στο τρέχον ερώτημα.
2. Μην επινοείς διαδικασίες, προθεσμίες, ποσά, δικαιολογητικά
   ή κανονιστικές διατάξεις.
3. Αν τα αποσπάσματα δεν επαρκούν, δήλωσε ξεκάθαρα τι δεν
   τεκμηριώνεται και τι πληροφορία λείπει.
4. Αν υπάρχουν αντικρουόμενες πληροφορίες, επισήμανέ το.
   Μην επιλέγεις αυθαίρετα μία εκδοχή.
5. Χρησιμοποίησε παραπομπές της μορφής [1], [2] κ.ο.κ.
   Οι αριθμοί αντιστοιχούν στα αποσπάσματα που σου δίνονται.
6. Μην κατασκευάζεις τίτλους εγγράφων, URLs ή αριθμούς παραπομπών.
7. Το ιστορικό συνομιλίας βοηθά στην κατανόηση του ερωτήματος,
   αλλά δεν αποτελεί τεκμηρίωση. Χρησιμοποίησε μόνο τα τρέχοντα
   αποσπάσματα για πραγματολογικούς ισχυρισμούς.
8. Μην εμφανίζεις εσωτερική συλλογιστική.

Δώσε μια σύντομη αλλά ουσιαστική απάντηση. Όπου είναι χρήσιμο,
χρησιμοποίησε αριθμημένα βήματα ή κουκκίδες.
"""

    user_prompt = f"""
Ερώτημα χρήστη:
{question}

Αποσπάσματα από τη βάση διοικητικών εγγράφων:

{build_context(passages)}

Σύνταξε την απάντηση στα ελληνικά, ακολουθώντας αυστηρά
τους παραπάνω κανόνες και προσθέτοντας παραπομπές στα
σχετικά αποσπάσματα.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_as_messages(history))
    messages.append({"role": "user", "content": user_prompt})

    return ollama_chat(messages)


# --------------------------------------------------
# Display retrieved sources
# --------------------------------------------------

def display_sources(passages):
    print("\n" + "=" * 72)
    print("ΠΗΓΕΣ / ΑΝΑΚΤΗΘΕΝΤΑ ΑΠΟΣΠΑΣΜΑΤΑ")
    print("=" * 72)

    if not passages:
        print("Δεν βρέθηκαν σχετικά αποσπάσματα.")
        return

    for passage in passages:
        print(f"\n[{passage['rank']}] {passage['document_title']}")
        print(f"    Ενότητα: {passage['section']}")
        print(
            f"    Γραμμές: {passage['line_start']} - "
            f"{passage['line_end']}"
        )

        if passage["source_url"]:
            print(f"    URL: {passage['source_url']}")

        print(f"    Chunk ID: {passage['chunk_id']}")


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    history = []

    print("\n" + "=" * 72)
    print("POLYMNIA – Greek Administrative Information Assistant")
    print("=" * 72)
    print(f"LLM: {OLLAMA_MODEL}")
    print(f"Retrieved passages per query: {N_RESULTS}")
    print(f"Conversation memory: {MAX_HISTORY_EXCHANGES} exchanges")
    print("\nΠληκτρολογήστε την ερώτησή σας.")
    print("Εντολές: exit / quit / reset / sources on / sources off")

    show_sources = True

    try:
        while True:
            question = input("\nΕσείς: ").strip()

            if not question:
                continue

            command = question.lower()

            if command in {"exit", "quit"}:
                break

            if command == "reset":
                history.clear()
                print("Το ιστορικό συνομιλίας διαγράφηκε.")
                continue

            if command == "sources on":
                show_sources = True
                print("Η εμφάνιση πηγών ενεργοποιήθηκε.")
                continue

            if command == "sources off":
                show_sources = False
                print("Η εμφάνιση πηγών απενεργοποιήθηκε.")
                continue

            try:
                print("\n[1/3] Επεξεργασία ερωτήματος...")
                retrieval_query = rewrite_question(question, history)

                if retrieval_query != question:
                    print(f"Ερώτημα αναζήτησης: {retrieval_query}")

                print("[2/3] Αναζήτηση στη βάση εγγράφων...")
                passages = search(retrieval_query, n_results=N_RESULTS)

                print(f"Ανακτήθηκαν {len(passages)} αποσπάσματα.")

                print("[3/3] Σύνθεση απάντησης από το Krikri...")
                answer = generate_answer(question, passages, history)

                print("\n" + "=" * 72)
                print("POLYMNIA")
                print("=" * 72)
                print(answer)

                if show_sources:
                    display_sources(passages)

                history.append({
                    "question": question,
                    "answer": answer,
                })

                history = history[-MAX_HISTORY_EXCHANGES:]

            except Exception as exc:
                print(f"\nΣφάλμα κατά την επεξεργασία: {exc}")
                print("Μπορείτε να δοκιμάσετε ξανά.")

    finally:
        print("\nClosing embedding model...")
        try:
            embedding_model.stop_self_pool()
        except Exception as exc:
            print(f"Cleanup warning: {exc}")

        print("Shutdown completed.")


if __name__ == "__main__":
    main()
