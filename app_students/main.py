from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field

from app_students.rag import initialize_retriever, search
from app_students.llm import rewrite_question, generate_answer


APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_retriever()
    yield
    print("Shutting down Polymnia Students...")


app = FastAPI(
    title="Polymnia - Student Assistant",
    description="RAG-based student information assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=APP_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=APP_DIR / "templates")


# --------------------------------------------------
# Homepage
# --------------------------------------------------

@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# --------------------------------------------------
# Chat API
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict] = Field(default_factory=list)


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        question = request.message.strip()

        if not question:
            raise HTTPException(status_code=400, detail="Empty question")

        # Accept only simple user/assistant messages and limit their size.
        history = [
            {
                "role": item["role"],
                "content": item["content"][:2000],
            }
            for item in request.history
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
        ][-12:]

        # 1. Resolve references in follow-up questions.
        search_query = rewrite_question(question, history)

        # 2. Retrieve relevant passages.
        retrieved = search(query=search_query, n_results=3)

        # 3. Generate a grounded answer.
        answer = generate_answer(
            question=question,
            retrieved=retrieved,
            history=history,
        )

        # 4. Return concise source metadata for display in the UI.
        sources_by_document = {}

        for item in retrieved:

            title = item.get("document_title") or "Άγνωστο έγγραφο"
            section = item.get("section") or ""

            if title not in sources_by_document:
                sources_by_document[title] = {
                    "title": title,
                    "sections": []
                }

            # Keep up to two distinct section titles per document.
            if (
                section
                and section not in sources_by_document[title]["sections"]
                and len(sources_by_document[title]["sections"]) < 2
            ):
                sources_by_document[title]["sections"].append(section)

        sources = list(sources_by_document.values())

        return {
            "answer": answer,
            "sources": sources,
            "search_query": search_query,
        }

    except HTTPException:
        raise

    except requests.RequestException as exc:
        print(f"LLM/embedding service error: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Δεν ήταν δυνατή η επικοινωνία με την υπηρεσία Ollama.",
        )

    except Exception as exc:
        print(f"Student chat error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Παρουσιάστηκε σφάλμα κατά τη δημιουργία της απάντησης.",
        )


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}
