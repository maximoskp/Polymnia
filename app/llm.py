
import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL_NAME = "ilsp/llama-krikri-8b-instruct"

NUM_CTX = 8192

KEEP_ALIVE = "5m"

TIMEOUT = 180


# --------------------------------------------------
# Ollama communication
# --------------------------------------------------

def ollama_chat(messages, temperature=0.2):

    payload = {

        "model": MODEL_NAME,

        "messages": messages,

        "stream": False,

        "keep_alive": KEEP_ALIVE,

        "options": {
            "temperature": temperature,
            "num_ctx": NUM_CTX
        }

    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"].strip()


# --------------------------------------------------
# Follow-up question rewriting
# --------------------------------------------------

def rewrite_question(question, history):

    if not history:
        return question

    conversation = []

    for message in history[-12:]:

        role = message.get("role")

        content = message.get("content", "")

        if role not in {"user", "assistant"}:
            continue

        if not content:
            continue

        conversation.append(
            f"{role.upper()}: {content[:2000]}"
        )

    history_text = "\n".join(conversation)

    messages = [

        {
            "role": "system",

            "content": """
You are a question reformulation assistant.

Your task is to rewrite the user's latest question
as a standalone question suitable for semantic
retrieval from Greek administrative documents.

Use the conversation history to resolve references
such as "this procedure", "those documents",
"and what about the deadlines?" etc.

Preserve the original intent.

Do not answer the question.

Return ONLY the rewritten question.
If the question is already self-contained,
return it unchanged.
"""
        },

        {
            "role": "user",

            "content": f"""
Conversation history:

{history_text}

Latest question:

{question}
"""
        }

    ]

    rewritten = ollama_chat(
        messages,
        temperature=0.0
    )

    return rewritten or question


# --------------------------------------------------
# Final answer generation
# --------------------------------------------------

def generate_answer(question, retrieved, history=None):

    context_parts = []

    for i, item in enumerate(retrieved):

        context_parts.append(
            f"""
[SOURCE {i+1}]

Document:
{item['document_title']}

Section:
{item['section']}

Content:
{item['text']}
"""
        )

    context = "\n\n".join(context_parts)

    history_text = ""

    if history:

        history_text = "\n".join(

            f"{m['role'].upper()}: {m['content'][:2000]}"

            for m in history[-12:]

            if m.get("role") in {"user", "assistant"}
        )

    system_prompt = """
Είσαι η Polymnia, ένας ψηφιακός βοηθός
του Ελληνικού Μεσογειακού Πανεπιστημίου.

Βοηθάς το προσωπικό του Πανεπιστημίου να
εντοπίζει και να κατανοεί πληροφορίες σχετικά
με τις διοικητικές διαδικασίες.

ΚΑΝΟΝΕΣ:

1. Απάντησε κατά προτίμηση στα ελληνικά,
   με σαφή και φυσικό τρόπο.

2. Βασίσου πρωτίστως στις πληροφορίες
   που περιλαμβάνονται στις παρεχόμενες πηγές.

3. Μην επινοείς διαδικασίες, προθεσμίες,
   δικαιολογητικά ή αρμοδιότητες.

4. Εάν οι πηγές δεν επαρκούν για να απαντήσεις,
   δήλωσε ξεκάθαρα ότι δεν βρήκες επαρκή
   πληροφορία στον διαθέσιμο οδηγό.

5. Εάν οι πηγές περιέχουν αντικρουόμενες
   πληροφορίες, επισήμανέ το.

6. Μπορείς να χρησιμοποιείς τη συνομιλία
   για να κατανοήσεις το ερώτημα, αλλά οι
   πραγματολογικές πληροφορίες πρέπει να
   τεκμηριώνονται από τις παρεχόμενες πηγές.

7. Όταν χρησιμοποιείς πληροφορίες από τις
   πηγές, βάλε αντίστοιχες παραπομπές
   όπως [SOURCE 1], [SOURCE 2].

8. Μην ισχυρίζεσαι ότι μια πληροφορία
   περιλαμβάνεται στον οδηγό εάν δεν
   υποστηρίζεται από τις πηγές.

9. Εάν το ερώτημα δεν αφορά τον οδηγό
   διοικητικών διαδικασιών, εξήγησε ευγενικά
   ότι ο ρόλος σου είναι να βοηθάς σε αυτό
   το συγκεκριμένο πεδίο.
"""

    user_prompt = f"""
ΙΣΤΟΡΙΚΟ ΣΥΝΟΜΙΛΙΑΣ:

{history_text}

ΔΙΑΘΕΣΙΜΕΣ ΠΗΓΕΣ:

{context}

ΕΡΩΤΗΣΗ ΧΡΗΣΤΗ:

{question}

Δώσε μια τεκμηριωμένη και κατανοητή απάντηση.
"""

    answer = ollama_chat(

        [
            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2

    )

    return answer
