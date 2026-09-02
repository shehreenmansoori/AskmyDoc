# Document Assistant — Doc Chatbot

A document chatbot with a custom Gradio UI: upload a PDF, ask questions about
it, and fall back to live web search when the answer isn't in your documents.

## Features

- **PDF upload & RAG** — PDFs are chunked, embedded, and stored in Qdrant; questions are answered from retrieved chunks with source citations.
- **Web search fallback** — when the answer isn't in the document, DuckDuckGo results are fetched, stored, and used to answer.
- **Small talk & tools** — greetings, date/time (IST), basic math, and translation are handled by a LangGraph router before hitting RAG.
- **Session memory** — chat history, sessions, and repeat-question caching are persisted in MongoDB.
- **Custom UI** — hand-built Gradio interface with dark mode, sidebar drawer, settings panel, message actions, and document management.

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Gradio |
| Orchestration | LangGraph |
| LLM | Mistral AI (`mistral-small-latest`) |
| Embeddings | Mistral AI embeddings |
| Vector store | Qdrant Cloud |
| Database | MongoDB Atlas |
| PDF parsing | PyMuPDF |
| Web search | DuckDuckGo |

## Project Structure

```
app.py         Gradio UI + LangGraph agent (entry point)
chunking.py    PDF → text chunks
embedding.py   Chunks → Qdrant vectors
retreiver.py   Qdrant retrievers (PDF + web collections)
mongo_db.py    MongoDB connection
main.py        FastAPI variant of the backend
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (or set environment variables) with:

```
MISTRAL_API_KEY=...
MONGODB_URI=...
QDRANT_URL=...
QDRANT_API_KEY=...
```

Run the app:

```bash
python app.py
```

The server binds to `0.0.0.0` on `$PORT` (default 7860), so it works locally
and on hosts like Render out of the box.

## Deploying on Render

The repo includes a `render.yaml` for one-click deployment. Create a Web
Service (free plan works), set the four environment variables above, and
deploy — build and start commands are already configured.
