# Polymnia – Student Chat

A separate FastAPI + Ollama + ChromaDB application for student-facing questions.

## 1. Configure the student vector database

By default, the application expects:

`<project-root>/data/students/vectordb/chroma`

and a ChromaDB collection named:

`hmu_student_documents`

Set environment variables if your database uses a different location or collection:

```bash
export STUDENTS_DB_DIR="/absolute/path/to/student/chroma"
export STUDENTS_COLLECTION_NAME="your_student_collection"
```

The collection should have been created with the same embedding model and dimensionality
used by this application (by default `bge-m3`, 1024 dimensions).

The retriever expects useful metadata fields such as:

- `document_title` (or `title` / `filename`)
- `section` (or `heading`)
- `document_id`

The text itself is retrieved from the ChromaDB document field.

## 2. Configure Ollama

The defaults reuse the models from the admin application:

```bash
export OLLAMA_HOST="http://localhost:11434"
export EMBEDDING_MODEL="bge-m3"
export EMBEDDING_DIM="1024"
export LLM_MODEL="ilsp/llama-krikri-8b-instruct"
```

Optional settings include `LLM_NUM_CTX`, `LLM_TIMEOUT`, `LLM_KEEP_ALIVE`,
`EMBEDDING_TIMEOUT`, and `EMBEDDING_KEEP_ALIVE`.

## 3. Run

Run from the Polymnia project root so that `app_students` is importable:

```bash
uvicorn app_students.main:app --host 0.0.0.0 --port 8002
```

Open `http://localhost:8002`.

## 4. Notes

- Chat conversations are stored in the user's browser using `localStorage`.
- The backend receives at most the last 12 previous messages for context.
- Source references display document titles and sections, without hyperlinks.
- The frontend displays model responses as plain text for safety. Markdown rendering
  can be added later if desired.
- This is a prototype: browser-local histories are not user accounts and are not
  synchronized across browsers or devices.
