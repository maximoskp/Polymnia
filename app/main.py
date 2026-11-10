
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse

from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.rag import initialize_retriever, search
from app.llm import rewrite_question, generate_answer

import requests

import re

# For adding chat to all article sub-pages
CHAT_WIDGET = """
<button id="polymnia-launcher">
    💬 Ρωτήστε την Polymnia
</button>

<div id="polymnia-chat">

    <div class="chat-header">
        <h3>Polymnia – Ψηφιακός Βοηθός</h3>
        <button id="polymnia-close">×</button>
    </div>

    <div id="polymnia-messages" class="chat-messages">
        <div class="chat-message assistant">
            Γεια σας! Είμαι η Polymnia.
            Πώς μπορώ να σας βοηθήσω;
        </div>
    </div>

    <div id="polymnia-status" class="chat-status"></div>

    <div class="chat-input-area">
        <textarea id="polymnia-input"
            placeholder="Γράψτε την ερώτησή σας..."></textarea>
        <button id="polymnia-send">➤</button>
    </div>

</div>

<link rel="stylesheet" href="/static/css/chat.css">
<script src="/static/js/chat.js" defer></script>
"""


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

ARTICLES_DIR = (
    BASE_DIR
    / "data"
    / "odigos_dioikitikon_diadikasion"
    / "articles"
)

APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):

    initialize_retriever()

    yield

    print("Shutting down Polymnia...")


# --------------------------------------------------
# Application
# --------------------------------------------------

app = FastAPI(
    title="Polymnia - HMU Administrative Guide",
    description="Demonstration of the HMU Administrative Guide",
    version="0.2.0",
    lifespan=lifespan
)

app.mount(
    "/static",
    StaticFiles(directory=APP_DIR / "static"),
    name="static"
)

templates = Jinja2Templates(
    directory=APP_DIR / "templates"
)


# --------------------------------------------------
# Article metadata
# --------------------------------------------------

def get_articles():

    articles = []

    for i in range(1, 35):

        article_id = f"{i:02d}"

        filename = f"arthro_{article_id}.html"

        filepath = ARTICLES_DIR / filename

        if not filepath.exists():
            continue

        articles.append({
            "id": article_id,
            "filename": filename,
            "title": f"Άρθρο {i}",
            "url": f"/articles/{article_id}"
        })

    return articles


# --------------------------------------------------
# Homepage
# --------------------------------------------------

@app.get("/")
async def homepage(request: Request):

    articles = get_articles()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "articles": articles
        }
    )


# --------------------------------------------------
# Individual articles
# --------------------------------------------------


@app.get("/articles/{article_id}")
async def article(article_id: str):

    if not article_id.isdigit():
        raise HTTPException(status_code=404)

    article_id = article_id.zfill(2)

    number = int(article_id)

    if not 1 <= number <= 34:
        raise HTTPException(status_code=404)

    filepath = ARTICLES_DIR / f"arthro_{article_id}.html"

    if not filepath.exists():
        raise HTTPException(status_code=404)

    html = filepath.read_text(encoding="utf-8")

    previous_link = (
        f'<a href="/articles/{number-1:02d}">← Προηγούμενο</a>'
        if number > 1 else ""
    )

    next_link = (
        f'<a href="/articles/{number+1:02d}">Επόμενο →</a>'
        if number < 34 else ""
    )

    navigation = f"""
    <div id="polymnia-demo-navigation">
        <a href="/">⌂ Αρχική σελίδα</a>
        <div>
            {previous_link}
            {next_link}
        </div>
    </div>

    <style>
        #polymnia-demo-navigation {{
            position: sticky;
            top: 0;
            z-index: 99999;
            background: #174b78;
            color: white;
            padding: 12px 5%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 15px;
            font-family: Arial, sans-serif;
        }}

        #polymnia-demo-navigation a {{
            color: white;
            text-decoration: none;
            margin-left: 15px;
            font-size: 14px;
        }}

        #polymnia-demo-navigation a:hover {{
            text-decoration: underline;
        }}
    </style>
    """

    # chat
    html = re.sub(
        r"</body\s*>",
        lambda match: (
            CHAT_WIDGET
            + navigation
            + match.group(0)
        ),
        html,
        count=1,
        flags=re.IGNORECASE
    )

    if "</body>" in html.lower():

        # Insert before the closing body tag.

        html = re.sub(
            r"</body\s*>",
            lambda match: navigation + match.group(0),
            html,
            count=1,
            flags=re.IGNORECASE
        )

    else:
        html += navigation

    return HTMLResponse(content=html)



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
            raise HTTPException(
                status_code=400,
                detail="Empty question"
            )

        # Keep the last 6 exchanges.
        # history = request.history[-12:]
        history = [
            {
                "role": m["role"],
                "content": m["content"][:2000]
            }
            for m in request.history
            if isinstance(m, dict)
            and m.get("role") in {"user", "assistant"}
            and isinstance(m.get("content"), str)
        ][-12:]

        # 1. Rewrite follow-up questions.
        search_query = rewrite_question(
            question,
            history
        )

        # 2. Retrieve relevant passages.
        retrieved = search(
            query=search_query,
            n_results=5
        )

        # 3. Generate the answer.
        answer = generate_answer(
            question=question,
            retrieved=retrieved,
            history=history
        )

        # 4. Prepare source references.
        sources = []

        for i, item in enumerate(retrieved):

            sources.append({

                "number": i + 1,

                "title": item["document_title"],

                "section": item["section"],

                "url": item["article_url"],

                "original_url": item["source_url"],

                "similarity": item["similarity"]

            })

        return {

            "answer": answer,

            "sources": sources,

            "search_query": search_query

        }
    except HTTPException:
        raise
    
    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=f"LLM service error: {str(e)}"
        )

    except Exception as e:

        print(f"Chat error: {e}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the answer."
        )


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "articles": len(get_articles())
    }
