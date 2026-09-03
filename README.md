<div align="center">

# DocuMind
### Multi-Tool LangGraph Document Intelligence & Hybrid RAG Engine

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=flat-square&logo=render&logoColor=black)](https://doc-chatbot-1.onrender.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Orchestration: LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph%20StateGraph-1C7C54?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Vector DB: Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%20Cloud-DC2626?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Persistence: MongoDB](https://img.shields.io/badge/Persistence-MongoDB%20Atlas-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![LLM: Mistral AI](https://img.shields.io/badge/LLM-Mistral%20Small-FD6F00?style=flat-square&logo=mistralai&logoColor=white)](https://mistral.ai/)
[![Web Fallback: DDG](https://img.shields.io/badge/Search-DuckDuckGo%20Engine-DE5833?style=flat-square)](https://duckduckgo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

<p align="center">
  <b>Hybrid document reasoning combining PyMuPDF chunking, dense vector retrieval, autonomous LangGraph tool routing, live web fallback, and MongoDB session persistence.</b>
</p>

[**Live Demo**](https://doc-chatbot-1.onrender.com/) | [**Architecture**](#state-machine-architecture) | [**Hybrid Retrieval**](#dual-tier-retrieval--fallback) | [**Engineering Specs**](#engineering-specifications) | [**Quickstart**](#quickstart)

</div>

---

## Problem & Architecture Overview

Standard RAG pipelines break down in two common production scenarios:
1. **The Out-of-Domain Dead End**: When a user asks a question not contained in the uploaded PDF, standard RAG either hallucinates or unhelpfully asserts *"I don't know"*.
2. **Context Bleed & Session Loss**: Stateless chat interfaces lose user intent, file context, and conversational continuity across turns.

**DocuMind** solves this with an autonomous **LangGraph StateGraph** architecture:
* **Dynamic Intent Routing**: Routes incoming requests to dedicated tool nodes (system time, arithmetic, general dialogue) without polluting the vector retrieval pipeline.
* **Dual-Tier Retrieval with Web Fallback**: If retrieved document chunks score below confidence thresholds, the engine dynamically triggers a DuckDuckGo live search, vectorizes the search results on the fly, and synthesizes a grounded answer.
* **Dual-Collection Vector Segmentation**: Isolates uploaded PDF embeddings from ephemeral web search vectors in Qdrant to eliminate cross-domain contamination.
* **MongoDB Session Memory**: Persists conversation history, active document metadata, and session tokens across client restarts.

---

## State Machine Architecture

```mermaid
flowchart TD
    subgraph Client["1. User & Ingestion Layer"]
        A[User Query or PDF Upload] --> B[Gradio Interface / REST API]
        B -->|PDF File| C[PyMuPDF + Recursive Splitter]
        C --> D[(Qdrant Cloud: PDF Collection)]
    end

    subgraph StateGraph["2. LangGraph Autonomous Router"]
        B -->|User Message| E{LangGraph State Router}
        E -->|Math / Tools| F[Deterministic Tool Nodes]
        E -->|Smalltalk / Direct| G[Direct LLM Synthesis]
        E -->|Document Question| H[Retrieve Node]
    end

    subgraph RAG["3. Dual-Tier Knowledge Resolution"]
        H -->|Cosine Search k=4| D
        D --> I{Context Found & Confident?}
        I -->|Yes| J[PDF Answer Node with Exact Citations]
        I -->|No / Low Similarity| K[Web Fallback Node]
        K --> L[DuckDuckGo Live Search API]
        L --> M[(Qdrant: Web Search Collection)]
        M --> N[Web-Grounded Answer Node]
    end

    subgraph Storage["4. State Persistence"]
        J & N & F & G --> O[Response Assembler]
        O --> P[(MongoDB Atlas: Session State)]
        O --> Q[Streamed UI Response]
    end

    style Client fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style StateGraph fill:#0f172a,stroke:#818cf8,stroke-width:2px,color:#fff
    style RAG fill:#0f172a,stroke:#f43f5e,stroke-width:2px,color:#fff
    style Storage fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#fff
```

---

## Dual-Tier Retrieval & Fallback

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Router as LangGraph State Router
    participant Qdrant as Qdrant Vector Cloud
    participant DDG as DuckDuckGo Live Search
    participant LLM as Mistral AI Engine
    participant Mongo as MongoDB Atlas

    User->>Router: "What are the Q3 projection numbers | "
    Router->>Qdrant: Query PDF Collection (dense embeddings)
    alt High Confidence Match
        Qdrant-->>Router: Top-4 document excerpts
        Router->>LLM: Synthesize answer with page-number citations
        LLM-->>User: "According to page 14, Q3 projected revenue is..."
    else Zero Match / Out of Domain
        Qdrant-->>Router: Insufficient context (< similarity threshold)
        Router->>DDG: Execute live web query
        DDG-->>Router: Real-time search snippets
        Router->>Qdrant: Index ephemeral web results
        Router->>LLM: Synthesize web-grounded answer + cite URLs
        LLM-->>User: "Document does not contain this. According to web sources..."
    end
    Router->>Mongo: Persist turn, citations, and intent metadata
```

---

## Engineering Specifications

| Layer | Technology | Architectural Role | :--- | :--- | :--- | **Agent Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Deterministic acyclic state routing with conditional branching | **Vector Engine** | [Qdrant Cloud](https://qdrant.tech/) | Production dense cosine similarity search (`mistral-embed`) | **Persistence** | [MongoDB Atlas](https://www.mongodb.com/) | User session history, conversation turns, and document logs | **Document Ingestion** | PyMuPDF (`fitz`) + Recursive Splitter | High-speed layout-aware PDF text and table extraction | **LLM Provider** | [Mistral AI](https://mistral.ai/) (`mistral-small-latest`) | Low-latency instruction-tuned reasoning and source citation | **Web Search Fallback**| `duckduckgo-search` | Live internet fallback when local document context is exhausted | **Frontend UI** | Gradio | Responsive UI with dark mode, drawer navigation, and settings |

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- Active accounts for Mistral AI, Qdrant Cloud, and MongoDB Atlas.

### 2. Clone & Install
```bash
git clone https://github.com/shehreenmansoori/Doc-Chatbot.git
cd Doc-Chatbot

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_api_key
QDRANT_URL=your_qdrant_cloud_endpoint
QDRANT_API_KEY=your_qdrant_api_key
MONGODB_URI=your_mongodb_atlas_connection_string
PORT=7860
```

### 4. Launch Service
```bash
python app.py
```
Access the application at [http://localhost:7860](http://localhost:7860).

---

## Cloud Deployment (Render / Docker)

The repository contains a native `render.yaml` configuration for automated deployment:
```yaml
services:
  - type: web
    name: doc-chatbot
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PORT
        value: 7860
```
The server dynamically binds to `0.0.0.0` on `$PORT` to ensure compatibility across cloud container runtimes.

---

## Repository Layout

```text
├── app.py           # Gradio application interface & LangGraph state machine
├── chunking.py      # PyMuPDF extraction & recursive character splitting
├── embedding.py     # Qdrant client connection & dense vector indexing
├── retreiver.py     # Multi-collection retrievers (PDF + Web collections)
├── mongo_db.py      # MongoDB Atlas session persistence & chat logging
├── main.py          # Alternative headless FastAPI backend endpoints
├── requirements.txt # Pinned production Python dependencies
└── render.yaml      # Render infrastructure-as-code deployment manifest
```

---

## License
Distributed under the [MIT License](LICENSE).





