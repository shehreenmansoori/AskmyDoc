"""
Gradio frontend for the Document Chatbot.

Provides:
  - PDF upload with processing feedback
  - Chat interface with conversation history
  - Session management (new / continue)
  - Web-search fallback toggle
  - Expandable source citations
"""

import gradio as gr
import uuid
import os
import re
import shutil
import logging
import threading
from datetime import datetime, timezone,timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

# Used to escape text coming from PDFs / web results before it is placed
# into HTML source cards (prevents markup injection).
from html import escape as html_escape

from chunking import process_pdf
from langchain.tools import tool
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from embedding import store_documents,store_web_documents
from retreiver import get_retreiver
from mongo_db import metadata_collection

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from qdrant_client import QdrantClient, models
from starlette.exceptions import StarletteDeprecationWarning
import warnings
from typing import TypedDict
from datetime import datetime
 
warnings.filterwarnings(
    "ignore",
    category=StarletteDeprecationWarning,
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)

#--Tools-----------------------------------------------------------------------
@tool
def date_time_tool(query: str) -> str:
    """Use this for date, time, day, today, tomorrow, yesterday, and year questions."""
    tz = ZoneInfo("Asia/Kolkata")
    now = datetime.now(tz)

    q = query.lower()

    if "time" in q:
        return now.strftime("Current time is %I:%M %p")

    if "tomorrow" in q:
        tomorrow = now + timedelta(days=1)
        return tomorrow.strftime("Tomorrow is %A, %d %B %Y")

    if "yesterday" in q:
        yesterday = now - timedelta(days=1)
        return yesterday.strftime("Yesterday was %A, %d %B %Y")

    if "year" in q:
        return f"Current year is {now.year}"

    if "today" in q or "date" in q:
        return now.strftime("Today is %A, %d %B %Y")

    if "day" in q:
        return now.strftime("Today is %A")
    return now.strftime("Current date and time is %A, %d %B %Y, %I:%M %p")

@tool
def simple_reply_tool(query:str)->str:
    """Use this tool when user writes a greeting"""
    q = query.lower().strip()

    greetings = ["hi", "hey", "hello", "yo", "sup"]
    thanks = ["thank you","ty","thanks"]

    if q in greetings:
        return "Hey! Upload a PDF or Ask me a question."
    
    if q in thanks:
        return "You're welcome!"
    return "I am not sure how respond to that, upload a pdf or try asking me a question."
    
@tool
def math_tool(query):
    """Use this tool to solve basic maths problems"""
    if re.fullmatch(r"[0-9+\-*/().\s]+",query):
        # Guard the evaluator: reject oversized inputs and exponent towers
        # like 9**9**9**9 that would otherwise hang the whole server.
        has_power = "**" in query
        safe_size = (
            len(query) <= 100
            and (not has_power or (query.count("**") == 1 and not re.search(r"\d{5,}", query)))
        )
        if safe_size:
            try:
                return str(eval(query, {"__builtins__":{}}))
            except Exception:
                return "Sorry, I couldn't evaluate that expression."
    return "Sorry, I couldn't evaluate that expression."

@tool
def translation_agent_tool(query: str) -> str:
    """Use this specialist LLM tool to translate user text."""
    response = base_llm.invoke(f"Extract the text and target language from this request, then translate it:\n{query}")
    return response.content

tools = [date_time_tool,simple_reply_tool,math_tool,translation_agent_tool]

# ── LLM ───────────────────────────────────────────────────────────

# temperature=0 keeps RAG answers deterministic — at the default temperature
# the model randomly replies NOT_FOUND even when the answer is in the context.
base_llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
tool_llm = base_llm.bind_tools(tools)

# ── Langraph ───────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    session_id: str
    enable_web_search: bool
    history_text: str
    normalized_query: str

    answer: str
    sources: list
    used_web_search: bool
    docs: list
    route: str

def route_query(state: AgentState):
    q = state["query"].lower().strip()

    if q in ["hi", "hello", "hey", "thanks", "thank you"]:
        logger.info("Route selected: simple")
        return {"route": "simple"}

    if re.fullmatch(r"[0-9+\-*/().\s]+", q):
        logger.info("Route selected: math")
        return {"route": "math"}
 
    # Whole-word matching so words like "sentiment" or "estimate" don't
    # accidentally contain "time" and get misrouted to the clock tool.
    if re.search(r"\b(date|time|today|tomorrow|yesterday|year|day)\b", q):
        logger.info("Route selected: date")
        return {"route": "date"}
    
    if any(word in q for word in ["translate","translation","translate this"]):
        logger.info("Route selected: translate")
        return {"route": "translate"}
    return {"route": "rag"}


def simple_node(state:AgentState):
    return{
        "answer": simple_reply_tool.invoke({"query":state["query"]}),
        "sources":[],
        "used_web_search": False
    }

def math_node(state:AgentState):
    return{
        "answer":math_tool.invoke({"query":state["query"]}),
        "sources":[],
        "used_web_search":False
    }

def date_node(state:AgentState):
    return{
        "answer":date_time_tool.invoke({"query":state["query"]}),
        "sources":[],
        "used_web_search":False
    } 

def translate_node(state: AgentState):
    return {
        "answer": translation_agent_tool.invoke({"query": state["query"]}),
        "sources": [],
        "used_web_search": False,
    }

def retrieve_node(state:AgentState):
    retreiver = get_retreiver()
    if retreiver is None:
        # No Qdrant collection yet (nothing uploaded) — answer from an
        # empty context instead of crashing the chat.
        logger.warning("No Qdrant collection available yet; skipping retrieval.")
        return {"docs": []}
    docs = retreiver.invoke(state["query"])
    logger.info(f"Retrieved documents: {len(docs)}")
    return {"docs":docs}
    
def pdf_node(state:AgentState):
    docs = state["docs"]
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = pdf_prompt.invoke({
        "history": state["history_text"],
        "context": context,
        "question": state["query"],
    })
    try:
        response = base_llm.invoke(final_prompt)
        logger.info("PDF answer generated successfully")
    except Exception as e:
        logger.exception("LLM failed")
        return{"answer":"Something went wrong while generating the answer."}
    answer = response.content
    
    sources = []
    if "NOT_FOUND" not in answer:
        sources = [
            {"source_number": i + 1, 
             "content": doc.page_content
            }
            for i, doc in enumerate(docs)
        ]
        logger.info(f"PDF answer found with sources : {len(sources)}")
    else:
        logger.info("Answer not found in PDF")

    return {
        "answer": answer,
        "sources": sources,
        "used_web_search": False,
    }

def should_web_search(state:AgentState)->Literal["save","web"]:
    if "NOT_FOUND" in state["answer"] and state["enable_web_search"]:
        logger.info("PDF answer not found. Routing to web search.")
        return "web"
    return "save"

def web_node(state:AgentState):
    try:
        search_tool = DuckDuckGoSearchResults(output_format="list")
        web_results = search_tool.invoke(state["query"])
    except Exception as e:
        # DuckDuckGo often rate-limits / blocks cloud provider IPs.
        logger.warning(f"Web search failed: {e}")
        return {
            "answer": (
                "Web search is unavailable right now "
                "(the search provider may be rate-limiting us). "
                "Please try again in a little while."
            ),
            "sources": [],
            "used_web_search": True,
        }

    store_web_documents(web_results,state["query"])

    final_prompt = web_prompt.invoke({
        "history":state["history_text"],
        "web_results": web_results,
        "question":state["query"]
    })
    response = base_llm.invoke(final_prompt)

    sources = [
        {
            "source_number": i,
            "title": result.get("title", ""),
            "url": result.get("link", ""),
            "content": result.get("snippet", ""),
        }
        for i, result in enumerate(web_results, start=1)
    ]
    return {
        "answer": response.content,
        "sources": sources,
        "used_web_search": True,
    }

def save_node(state:AgentState):
    answer = state["answer"]

    if not answer:
        answer = "No answer generated."
  
    if "NOT_FOUND" in answer:
        answer = (
            "I couldn't find the answer in the uploaded document. "
            "Turn on Web search to search the web for an answer."
        )

    metadata_collection.insert_one({
        "query": state["query"],
        "normalized_query": state["normalized_query"],
        "session_id": state["session_id"],
        "answer": answer,
        "used_web_search": state["used_web_search"],
        "timestamp": datetime.now(timezone.utc),
        "documents_retrieved": len(state.get("docs", [])),
        "sources": state["sources"],
    })
    return {"answer": answer}

def decide_route(state: AgentState):
    return state["route"]

#----Graph-------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("route_query",route_query)
graph.add_node("simple",simple_node)
graph.add_node("math",math_node)
graph.add_node("date",date_node)
graph.add_node("translate", translate_node)
graph.add_node("retrieve",retrieve_node)
graph.add_node("pdf",pdf_node)
graph.add_node("web",web_node)
graph.add_node("save",save_node)

graph.add_edge(START,"route_query")

graph.add_conditional_edges(
    "route_query",
    decide_route,
    {
        "simple": "simple",
        "math": "math",
        "date": "date",
        "translate": "translate",
        "rag": "retrieve",
    }
)
graph.add_edge("simple", "save")
graph.add_edge("math", "save")
graph.add_edge("date", "save")
graph.add_edge("translate", "save")

graph.add_edge("retrieve", "pdf")

graph.add_conditional_edges(
    "pdf",
    should_web_search,
    {
        "web": "web",
        "save": "save",
    }
)

graph.add_edge("web", "save")
graph.add_edge("save", END)

chat_graph = graph.compile()

#----------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

pdf_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Use the previous conversation and the provided context to answer the question.

If the answer is NOT contained in the context, reply with exactly:
NOT_FOUND
(nothing else, no explanation)

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:
""")

web_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

The answer was not found in the uploaded document, so you were given some
web search results instead. Use them (and the chat history) to answer the
question as best you can. Mention that the answer is based on a web search.

Chat History:
{history}

Web Search Results:
{web_results}

Question:
{question}

Answer:
""")

judge_prompt = ChatPromptTemplate.from_template("""
You are checking whether a previous answer is still good enough to be reused
for the exact same question, asked again by the same user in the same session.

Question:
{question}

Previous Answer:
{previous_answer}

Is the previous answer sufficient, complete, and correct enough to reuse as-is?
Reply with exactly one word: YES or NO.
""")

# ── Icons (inline SVG, styled via CSS `color`) ───────────────────────────────
DOC_ICON_SVG = (
    "<svg viewBox='0 0 24 24' width='18' height='18' fill='none' "
    "stroke='currentColor' stroke-width='1.6' stroke-linecap='round' "
    "stroke-linejoin='round'><path d='M7 3h7l4 4v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z'/>"
    "<path d='M14 3v4h4'/><path d='M9 13h6'/><path d='M9 17h6'/></svg>"
)

LOGO_MARK_HTML = (
    "<div class='logo-mark'>"
    "<svg viewBox='0 0 24 24' width='16' height='16' fill='none' stroke='#FFFFFF' "
    "stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M5 13l4 4L19 7'/></svg>"
    "</div>"
)

RECENT_SESSION_LIMIT = 12
SETTINGS_DOC_LIMIT = 12
ACTIVE_CHAT_REQUESTS = {}
ACTIVE_CHAT_REQUESTS_LOCK = threading.Lock()

def normalize_query(text: str) -> str:
    """
    Normalize a question for cache matching so trivial differences
    (case, punctuation, extra spaces) don't count as a "different" question.
    e.g. "What is X?" and "what is x" both become "what is x".
    This is plain string cleanup - no NLP or embeddings involved.
    """
    text = (text or "").lower().strip()
    text = re.sub(r"[?!.,;:'\"]+$", "", text)   # drop trailing punctuation
    text = re.sub(r"\s+", " ", text)            # collapse repeated whitespace
    return text.strip()

def get_uploaded_pdf_files():
    """Return uploaded PDF filenames."""
    os.makedirs("uploads", exist_ok=True)
    return sorted([f for f in os.listdir("uploads") if f.lower().endswith(".pdf")])


def get_recent_sessions():
    """Query MongoDB to retrieve recent chat sessions."""
    try:
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {
                "$group": {
                    "_id": "$session_id",
                    "latest_timestamp": {"$first": "$timestamp"},
                    "first_query": {"$last": "$query"},
                    "session_title": {"$max": "$session_title"},
                }
            },
            {"$sort": {"latest_timestamp": -1}},
            {"$limit": RECENT_SESSION_LIMIT}
        ]
        sessions = list(metadata_collection.aggregate(pipeline))
        return sessions
    except Exception as e:
        logger.error(f"Error fetching recent sessions: {e}")
        return []


def format_answer(answer: str, used_web_search: bool, sources: list) -> str:
    """Append a web-search note and a collapsible source list to an answer."""
    if used_web_search:
        answer += "\n\n<div class='source-note'>Answer based on a web search</div>"

    if sources:
        answer += "\n\n<details class='sources-panel'><summary>Sources</summary>\n\n"
        for s in sources:
            if s.get("url"):
                url = str(s.get("url") or "")
                # Only render real http(s) links; anything else is shown as text.
                if url.lower().startswith(("http://", "https://")):
                    safe_url = html_escape(url, quote=True)
                    link_html = (
                        f"<a class='source-link' href='{safe_url}' target='_blank' rel='noopener'>{safe_url}</a>"
                    )
                else:
                    link_html = f"<div class='source-snippet'>{html_escape(url)}</div>"
                answer += (
                    "<div class='source-item'>"
                    f"<span class='source-index'>{s['source_number']:02d}</span>"
                    "<div class='source-body'>"
                    f"<div class='source-title'>{html_escape(str(s.get('title') or 'Web result'))}</div>"
                    f"{link_html}"
                    f"<div class='source-snippet'>{html_escape(str(s.get('content', '')))}</div>"
                    "</div></div>\n\n"
                )
            else:
                content = str(s.get("content", ""))
                snippet = content[:300] + ("…" if len(content) > 300 else "")
                answer += (
                    "<div class='source-item'>"
                    f"<span class='source-index'>{s['source_number']:02d}</span>"
                    "<div class='source-body'>"
                    "<div class='source-title'>Document excerpt</div>"
                    f"<div class='source-snippet'>{html_escape(snippet)}</div>"
                    "</div></div>\n\n"
                )
        answer += "</details>"

    return answer


def load_session_history(session_id):
    """Load and format the chat history for a given session ID."""
    if not session_id:
        return []
    try:
        chats = list(
            metadata_collection.find({"session_id": session_id})
            .sort("timestamp", 1)
        )
        history = []
        for chat in chats:
            history.append({"role": "user", "content": chat["query"]})
            answer = format_answer(
                chat["answer"],
                chat.get("used_web_search", False),
                chat.get("sources", []),
            )
            history.append({"role": "assistant", "content": answer})
        return history
    except Exception as e:
        logger.error(f"Error loading session history: {e}")
        return []


def get_uploaded_documents_html():
    """Scan the uploads/ folder and generate a clean HTML card list of uploaded PDFs."""
    files = get_uploaded_pdf_files()
    if not files:   
        return "<div class='no-docs-message'>No documents uploaded yet.</div>"

    html = "<div class='uploaded-docs-list'>"
    for file in files:
        safe_file = html_escape(file, quote=True)
        html += f"""
        <div class='doc-card'>
            <div class='doc-card-icon'>{DOC_ICON_SVG}</div>
            <div class='doc-card-content'>
                <div class='doc-card-title' title='{safe_file}'>{safe_file}</div>
                <div class='doc-card-subtitle'>PDF document</div>
            </div>
        </div>
        """
    html += "</div>"
    return html


def upload_pdf(file):
    """Process an uploaded PDF: copy to local uploads/ → chunk → embed → store in Qdrant."""
    if file is None:
        return "*Upload a PDF to get started.*", get_uploaded_documents_html()
    
    # Gradio's UploadButton(type="filepath") hands us a local path string,
    # so accept both a plain path and a file-like object.
    file_name = file if isinstance(file, str) else getattr(file, "name", "")
    if not file_name.lower().endswith(".pdf"):
        return "Only PDF files are allowed.", get_uploaded_documents_html()

    try:
        os.makedirs("uploads", exist_ok=True)
        # `file` is already a local filepath string (gr.File(type="filepath")),
        # not a file-like object, so it has no `.name` attribute.
        dest_path = os.path.join("uploads", os.path.basename(file))

        # Copy file to uploads folder so we can track it
        shutil.copy(file, dest_path)
        logger.info(f"FILE COPIED TO UPLOADS: {dest_path}")

        with open(dest_path, "rb") as f:
            pdf_bytes = f.read()

        documents = process_pdf(pdf_bytes)
        for document in documents:
            document.metadata = {
                **document.metadata,
                "filename": os.path.basename(dest_path),
                "source": os.path.basename(dest_path),
            }
        logger.info(f"CHUNKS CREATED: {len(documents)}")

        count = store_documents(documents)
        logger.info(f"EMBEDDINGS CREATED: {count}")

        status_msg = (
            f"**Processed successfully**\n"
            f"- Chunks created: {len(documents)}\n"
            f"- Embeddings stored: {count}"
        )
        return status_msg, get_uploaded_documents_html()
    except Exception as e:
        logger.exception("Error processing PDF")
        return f"**Error processing file:** {str(e)}", get_uploaded_documents_html()


def _next_chat_request_id(session_id):
    request_id = str(uuid.uuid4())
    with ACTIVE_CHAT_REQUESTS_LOCK:
        ACTIVE_CHAT_REQUESTS[session_id] = request_id
    return request_id


def _is_active_chat_request(session_id, request_id):
    with ACTIVE_CHAT_REQUESTS_LOCK:
        return ACTIVE_CHAT_REQUESTS.get(session_id) == request_id


def _clear_chat_request(session_id, request_id):
    with ACTIVE_CHAT_REQUESTS_LOCK:
        if ACTIVE_CHAT_REQUESTS.get(session_id) == request_id:
            ACTIVE_CHAT_REQUESTS.pop(session_id, None)


def begin_ask(query, chat_history, session_id, web_search_enabled=False):
    """Immediately reflect the submitted question and clear the textbox."""
    query = (query or "").strip()
    chat_history = list(chat_history or [])
    if not query:
        return chat_history, "", "", ""

    if not session_id:
        session_id = str(uuid.uuid4())

    request_id = _next_chat_request_id(session_id)
    chat_history.append({
        "role": "user", 
        "content": query  
    })
    return chat_history, "", query, request_id


def delete_chat_message(chat_history, message_index):
    """Remove one visible message without affecting the rest of the conversation."""
    messages = list(chat_history or [])
    try:
        index = int(message_index)
    except (TypeError, ValueError):
        return messages

    if 0 <= index < len(messages):
        messages.pop(index)
    return messages
    

def ask(query, chat_history, session_id, enable_web_search, request_id):
    """
    Handle a user question:
      1. Retrieve relevant chunks from Qdrant
      2. Ask the LLM
      3. Optionally fall back to web search
      4. Persist to MongoDB
      5. Return updated Gradio chat history
    """
    query = (query or "").strip()
    if not query or not request_id:
        yield chat_history
        return

    if not session_id:
        session_id = str(uuid.uuid4())

    if not _is_active_chat_request(session_id, request_id):
        yield chat_history
        return

    # Fetch previous turns from MongoDB
    previous_chats = list(
        metadata_collection.find({
            "session_id": session_id
        }).sort("timestamp", -1).limit(3)
    )

    history_text = ""
    for chat in reversed(previous_chats):
        history_text += f"User: {chat['query']}\nBot: {chat['answer']}\n"

    # Check if this exact question was already asked in this session
    normalized_query = normalize_query(query)
    repeat_chat = metadata_collection.find_one({
        "session_id": session_id,
        "normalized_query": normalized_query
        },
        sort=[("timestamp", -1)]
    )
    logger.info(f"Repeat-question lookup: {'FOUND' if repeat_chat else 'NOT FOUND'}")

    if repeat_chat:
        logger.info("Repeat question matched. Reusing cached answer directly.")
        answer = format_answer(
            repeat_chat["answer"],
            repeat_chat.get("used_web_search", False),
            repeat_chat.get("sources", []),
        )

        metadata_collection.insert_one({
            "query":query,
            "normalized_query": normalized_query,
            "session_id":session_id,
            "answer":repeat_chat["answer"],
            "timestamp": datetime.now(timezone.utc),
            "used_web_search":repeat_chat.get("used_web_search",False),
            "sources": repeat_chat.get("sources",[])
        })
        chat_history = list(chat_history or [])
        chat_history.append({
            "role": "assistant",
            "content": answer
        })

        _clear_chat_request(session_id, request_id)
        yield chat_history
        return
    
    result = chat_graph.invoke({
        "query": query,
        "session_id": session_id,
        "enable_web_search": enable_web_search,
        "history_text": history_text,
        "normalized_query": normalized_query,
        "answer": "",
        "sources": [],
        "used_web_search": False,
        "docs": [],
        "route": "",
    })

    answer = format_answer(
        result["answer"],
        result.get("used_web_search", False),
        result.get("sources", []),
    )

    chat_history = list(chat_history or [])
    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    _clear_chat_request(session_id, request_id)
    yield chat_history
    return

def new_session():
    """Create a fresh session ID and clear the chat."""
    return str(uuid.uuid4()), [], False


def remember_web_search_setting(enabled, session_id, session_settings):
    """Keep the Web Search preference scoped to the active conversation."""
    settings = dict(session_settings or {})
    if session_id:
        settings[str(session_id)] = bool(enabled)
    return settings


# ── Helper functions for dynamic UI ──────────────────────────────────────────

def _session_button_updates(sessions, active_session_id=None):
    """Build the gr.Button updates for the recent-session slots in the sidebar."""
    updates = []
    for i in range(RECENT_SESSION_LIMIT):
        if i < len(sessions):
            label = sessions[i].get("session_title") or sessions[i].get("first_query") or "New chat"
            if len(label) > 28:
                label = label[:25] + "..."
            is_active = sessions[i].get("_id") == active_session_id
            updates.append(gr.update(value=label, visible=True, variant="primary" if is_active else "secondary"))
        else:
            updates.append(gr.update(visible=False))
    return updates


def on_app_load(session_id):
    """Initializes the app: loads recent sessions, documents, and sets the active session."""
    if not session_id:
        session_id = str(uuid.uuid4())
    sessions = get_recent_sessions()
    doc_html = get_uploaded_documents_html()
    empty_state = gr.update(visible=not bool(sessions))
    return [sessions, doc_html, empty_state] + _session_button_updates(sessions, session_id) + [session_id, session_id]

def refresh_sessions_list(existing_sessions=None, active_session_id=None):
    """Queries latest sessions from MongoDB and returns button updates."""
    sessions = get_recent_sessions()
    if not sessions and existing_sessions:
        sessions = existing_sessions
    empty_state = gr.update(visible=not bool(sessions))
    return [sessions, empty_state] + _session_button_updates(sessions, active_session_id)


def refresh_sessions_after_send(session_id, existing_sessions=None):
    """Refresh recent sessions without blanking the sidebar after a send."""
    sessions = get_recent_sessions()
    if not sessions:
        sessions = existing_sessions or []

    if session_id and not any(s.get("_id") == session_id for s in sessions):
        sessions = [{"_id": session_id, "first_query": "Current chat"}] + sessions

    sessions = sessions[:RECENT_SESSION_LIMIT]
    empty_state = gr.update(visible=not bool(sessions))
    return [sessions, empty_state] + _session_button_updates(sessions, session_id)


def _resolve_session_action_index(index, sessions):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    return index if 0 <= index < len(sessions or []) else None


def rename_conversation(index, title, recent_sessions, active_session_id):
    """Persist a custom conversation title and refresh the matching sidebar row."""
    sessions = [dict(session) for session in (recent_sessions or [])]
    resolved_index = _resolve_session_action_index(index, sessions)
    title = (title or "").strip()[:80]
    if resolved_index is None or not title:
        return [sessions, gr.update(visible=not bool(sessions))] + _session_button_updates(
            sessions, active_session_id
        )

    session_id = sessions[resolved_index].get("_id")
    try:
        metadata_collection.update_many(
            {"session_id": session_id},
            {"$set": {"session_title": title}},
        )
        sessions[resolved_index]["session_title"] = title
    except Exception as error:
        logger.error(f"Error renaming conversation {session_id}: {error}")

    return [sessions, gr.update(visible=not bool(sessions))] + _session_button_updates(
        sessions, active_session_id
    )


def delete_conversation(index, recent_sessions, active_session_id, web_search_settings):
    """Delete one conversation and select the next available session when needed."""
    sessions = [dict(session) for session in (recent_sessions or [])]
    search_settings = dict(web_search_settings or {})
    current_search_enabled = bool(search_settings.get(str(active_session_id), False))
    resolved_index = _resolve_session_action_index(index, sessions)
    if resolved_index is None:
        return [
            sessions,
            active_session_id,
            active_session_id,
            gr.update(),
            current_search_enabled,
            search_settings,
            gr.update(visible=not bool(sessions)),
        ] + _session_button_updates(sessions, active_session_id)

    deleted_session_id = sessions[resolved_index].get("_id")
    try:
        metadata_collection.delete_many({"session_id": deleted_session_id})
    except Exception as error:
        logger.error(f"Error deleting conversation {deleted_session_id}: {error}")
        return [
            sessions,
            active_session_id,
            active_session_id,
            gr.update(),
            current_search_enabled,
            search_settings,
            gr.update(visible=not bool(sessions)),
        ] + _session_button_updates(sessions, active_session_id)

    remaining_sessions = sessions[:resolved_index] + sessions[resolved_index + 1:]
    search_settings.pop(str(deleted_session_id), None)
    with ACTIVE_CHAT_REQUESTS_LOCK:
        ACTIVE_CHAT_REQUESTS.pop(deleted_session_id, None)

    if active_session_id == deleted_session_id:
        if remaining_sessions:
            next_index = min(resolved_index, len(remaining_sessions) - 1)
            next_session_id = remaining_sessions[next_index]["_id"]
            next_history = load_session_history(next_session_id)
        else:
            next_session_id = str(uuid.uuid4())
            next_history = []
    else:
        next_session_id = active_session_id
        next_history = gr.update()

    next_search_enabled = bool(search_settings.get(str(next_session_id), False))
    return [
        remaining_sessions,
        next_session_id,
        next_session_id,
        next_history,
        next_search_enabled,
        search_settings,
        gr.update(visible=not bool(remaining_sessions)),
    ] + _session_button_updates(remaining_sessions, next_session_id)


def _settings_document_updates():
    """Build document row updates for the settings panel."""
    files = get_uploaded_pdf_files()
    updates = [gr.update(visible=not bool(files))]
    for i in range(SETTINGS_DOC_LIMIT):
        if i < len(files):
            updates.extend([
                gr.update(visible=True),
                gr.update(value=files[i], visible=True),
                gr.update(visible=True),
            ])
        else:
            updates.extend([
                gr.update(visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
            ])
    return updates


def open_settings_panel():
    return [
        gr.update(visible=True),
        gr.update(value=""),
        gr.update(visible=False),
        None,
        gr.update(value="Delete this PDF?"),
    ] + _settings_document_updates()


def close_settings_panel():
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        None,
        gr.update(value=""),
        gr.update(value="Delete this PDF?"),
    )


def prepare_delete_document(index):
    files = get_uploaded_pdf_files()
    if index >= len(files):
        return gr.update(visible=False), None, "Document not found.", "Delete this PDF?"
    filename = files[index]
    return gr.update(visible=True), filename, "", f"Delete **{filename}**? This cannot be undone."


def ensure_qdrant_delete_indexes(client):
    """Create payload indexes needed for filtered deletes."""
    for field_name in ("metadata.filename", "metadata.source"):
        try:
            client.create_payload_index(
                collection_name="documents",
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Could not create Qdrant payload index for {field_name}: {e}")


def delete_document(filename):
    if not filename:
        return [
            gr.update(visible=False),
            None,
            "No document selected.",
            get_uploaded_documents_html(),
        ] + _settings_document_updates()

    safe_name = os.path.basename(filename)
    path = os.path.join("uploads", safe_name)
    try:
        if os.path.exists(path):
            os.remove(path)

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        ensure_qdrant_delete_indexes(client)
        delete_filter = models.Filter(
            should=[
                models.FieldCondition(key="metadata.filename", match=models.MatchValue(value=safe_name)),
                models.FieldCondition(key="metadata.source", match=models.MatchValue(value=safe_name)),
            ]
        )
        client.delete(
            collection_name="documents",
            points_selector=models.FilterSelector(filter=delete_filter),
            wait=True,
        )
        message = f"Deleted {safe_name}."
    except Exception as e:
        logger.exception("Error deleting document")
        message = f"Error deleting {safe_name}: {e}"

    return [
        gr.update(visible=False),
        None,
        message,
        get_uploaded_documents_html(),
    ] + _settings_document_updates()


def make_session_click_handler(index):
    """Factory for a recent-session button click handler bound to a fixed slot index."""
    def _handler(recent_sessions, web_search_settings):
        if len(recent_sessions) > index:
            sid = recent_sessions[index]["_id"]
            enabled = bool((web_search_settings or {}).get(str(sid), False))
            return sid, sid, load_session_history(sid), enabled
        return gr.update(), gr.update(), gr.update(), gr.update()
    return _handler


# ── Gradio UI ────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#FFFFFF",
    body_background_fill_dark="#111827",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#1F2937",
    block_border_width="1px",
    block_border_color="#E3E6EB",
    block_border_color_dark="#374151",
    block_label_text_color="#5B6472",
    block_label_text_color_dark="#D1D5DB",
    block_title_text_color="#16202E",
    block_title_text_color_dark="#F3F4F6",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#1F2937",
    input_border_color="#E3E6EB",
    input_border_color_dark="#374151",
    input_placeholder_color="#8B93A1",
    input_placeholder_color_dark="#9CA3AF",

    # Primary buttons (Navy)
    button_primary_background_fill="#16233F",
    button_primary_background_fill_dark="#2F5BEA",
    button_primary_background_fill_hover="#223457",
    button_primary_background_fill_hover_dark="#4472F2",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_primary_border_color="#16233F",
    button_primary_border_color_dark="#2F5BEA",

    # Secondary buttons (Border only)
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_dark="transparent",
    button_secondary_background_fill_hover="#F1F3F6",
    button_secondary_background_fill_hover_dark="#1F2937",
    button_secondary_text_color="#5B6472",
    button_secondary_text_color_dark="#D1D5DB",
    button_secondary_border_color="#E3E6EB",
    button_secondary_border_color_dark="#374151",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&display=swap');

/* ── Design tokens ────────────────────────────────────────────────── */
:root {
    --bg-page: #FFFFFF;
    --bg-sidebar: #F7F8FA;
    --bg-subtle: #F1F3F6;
    --bg-card: #FFFFFF;
    --bg-input: #FFFFFF;
    --bg-hover: #F9FAFB;
    --border: #E3E6EB;
    --border-strong: #D7DBE1;
    --border-hover: #D7DBE1;
    --navy: #16233F;
    --navy-hover: #223457;
    --accent: #2F5BEA;
    --accent-hover: #4472F2;
    --primary: #2F5BEA;
    --primary-hover: #2550D8;
    --link: #2F5BEA;
    --user-message: #16233F;
    --assistant-message: #FFFFFF;
    --text-primary: #16202E;
    --text-secondary: #5B6472;
    --text-muted: #8B93A1;
    --text-on-primary: #FFFFFF;
    --danger: #DC2626;
    --danger-hover: #B91C1C;
    --danger-subtle: #FEF2F2;
    --danger-border: #FCA5A5;
    --shadow-popover: 0 18px 45px rgba(15, 23, 42, 0.16);
    --shadow-soft: 0 1px 3px rgba(15, 23, 42, 0.04);
    --focus-ring: rgba(47, 91, 234, 0.12);
}

.dark-mode {
    --bg-page: #111827;
    --bg-sidebar: #161B22;
    --bg-subtle: #1F2937;
    --bg-card: #1F2937;
    --bg-input: #1F2937;
    --bg-hover: #263244;
    --border: #374151;
    --border-strong: #4B5563;
    --border-hover: #4B5563;
    --navy: #2F5BEA;
    --navy-hover: #4472F2;
    --primary: #3B82F6;
    --primary-hover: #2563EB;
    --link: #60A5FA;
    --user-message: #2563EB;
    --assistant-message: #1F2937;
    --text-primary: #F3F4F6;
    --text-secondary: #D1D5DB;
    --text-muted: #9CA3AF;
    --text-on-primary: #FFFFFF;
    --danger: #DC2626;
    --danger-hover: #B91C1C;
    --danger-subtle: rgba(220, 38, 38, 0.14);
    --danger-border: #DC2626;
    --shadow-popover: 0 18px 45px rgba(0, 0, 0, 0.35);
    --shadow-soft: none;
    --focus-ring: rgba(47, 91, 234, 0.25);
}

/* ── Global Styles ────────────────────────────────────────────────── */
/* Tailwind-style utilities used by generated Gradio status elements. */
.bg-transparent {
    background-color: transparent !important;
}
.text-black {
    color: #000000 !important;
}
.text-slate-500 {
    color: #64748B;
}
.dark-mode .dark\\:text-slate-400 {
    color: #94A3B8;
}

* {
    box-sizing: border-box;
    transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
}

body, html {
    background-color: var(--bg-page) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
    margin: 0;
    padding: 0;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background-color: var(--bg-page) !important;
}

.gradio-container .main.fillable {
    width: 100% !important;
    height: 100dvh !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gradio-container *,
.block,
.form,
.panel,
.contain,
.wrap,
.label-wrap {
    border-color: var(--border) !important;
}

.block,
.form,
.panel,
.contain {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text-primary) !important;
}

label,
.label-wrap,
.prose,
.prose * {
    color: var(--text-secondary) !important;
}

textarea,
input {
    background-color: transparent !important;
    color: var(--text-primary) !important;
}

::placeholder {
    color: var(--text-muted) !important;
    opacity: 1 !important;
}

button {
    border-color: var(--border) !important;
}

/* ── Two-Column Layout Grid ───────────────────────────────────────── */
.main-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    height: 100vh !important;
    gap: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}

/* ── Sidebar (280px width) ─────────────────────────────────────────── */
.sidebar-column {
    position: fixed !important;
    inset: 0 auto 0 0 !important;
    z-index: 1000 !important;
    width: 18rem !important;
    min-width: 0 !important;
    max-width: 85vw !important;
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100dvh !important;
    padding: 18px 14px !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    overscroll-behavior: contain;
    gap: 12px !important;
    justify-content: flex-start !important;
    transform: translateX(-100%) !important;
    visibility: hidden;
    transition: transform 0.22s ease, visibility 0.22s step-end;
    will-change: transform;
}

html.sidebar-open,
body.sidebar-open {
    overflow: hidden !important;
}

.sidebar-column.is-open,
body.sidebar-open .sidebar-column {
    transform: translateX(0) !important;
    visibility: visible;
    transition: transform 0.22s ease, visibility 0s step-start;
}

.sidebar-drawer-backdrop {
    position: fixed;
    inset: 0;
    z-index: 999;
    border: 0;
    background: rgba(15, 23, 42, 0.38);
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    transition: opacity 0.22s ease, visibility 0.22s step-end;
}

body.dark-mode .sidebar-drawer-backdrop {
    background: rgba(0, 0, 0, 0.56);
}

body.sidebar-open .sidebar-drawer-backdrop {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
    transition: opacity 0.22s ease, visibility 0s step-start;
}

@media (prefers-reduced-motion: reduce) {
    .sidebar-column,
    .sidebar-drawer-backdrop {
        transition-duration: 0.01ms !important;
    }
}

.drawer-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 8px !important;
    flex: 0 0 auto !important;
    min-width: 0 !important;
}

.drawer-header > div:first-child {
    min-width: 0 !important;
    flex: 1 1 auto !important;
}

#sidebar_close_btn,
#sidebar_open_btn {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 32px !important;
    min-width: 32px !important;
    max-width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 6px !important;
    background: transparent !important;
    color: var(--text-secondary) !important;
    box-shadow: none !important;
    font-size: 0 !important;
}

#sidebar_close_btn:hover,
#sidebar_close_btn:focus-visible,
#sidebar_open_btn:hover,
#sidebar_open_btn:focus-visible {
    background: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
    outline: none !important;
}

#sidebar_close_btn svg,
#sidebar_open_btn svg {
    width: 17px;
    height: 17px;
    stroke-width: 1.8;
}

@media (min-width: 768px) {
    .sidebar-column {
        position: relative !important;
        inset: auto !important;
        z-index: 1 !important;
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
        flex: 0 0 280px !important;
        height: 100dvh !important;
        overflow: hidden !important;
        transform: none !important;
        visibility: visible !important;
        transition: width 0.2s ease, min-width 0.2s ease, max-width 0.2s ease,
            flex-basis 0.2s ease, padding 0.2s ease !important;
        will-change: width;
    }

    .sidebar-column.desktop-collapsed {
        width: 52px !important;
        min-width: 52px !important;
        max-width: 52px !important;
        flex-basis: 52px !important;
        padding: 18px 10px !important;
    }

    .sidebar-column.desktop-collapsed > *:not(.drawer-header) {
        display: none !important;
    }

    .sidebar-column.desktop-collapsed #new_session_btn,
    .sidebar-column.desktop-collapsed .recent-section,
    .sidebar-column.desktop-collapsed .sidebar-footer,
    .sidebar-column.desktop-collapsed #document_list_html,
    .sidebar-column.desktop-collapsed #upload_status {
        display: none !important;
    }

    .sidebar-column.desktop-collapsed .drawer-header > div:first-child {
        display: none !important;
    }

    .sidebar-column.desktop-collapsed .drawer-header {
        width: 32px !important;
        justify-content: center !important;
    }

    #sidebar_open_btn {
        display: none !important;
    }

    .sidebar-drawer-backdrop {
        display: none !important;
    }

    .chat-column {
        transition: width 0.2s ease, flex-basis 0.2s ease;
    }
}

/* App Logo & Title */
.logo-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0;
}
.logo-mark {
    width: 30px;
    height: 30px;
    min-width: 30px;
    border-radius: 8px;
    background: var(--primary);
    display: flex;
    align-items: center;
    justify-content: center;
}
.logo-title {
    font-family: 'Manrope', 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}

/* "New Session" Button */
#new_session_btn {
    width: 100% !important;
    background-color: var(--primary) !important;
    color: var(--text-on-primary) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    text-align: center !important;
    justify-content: center !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    min-height: 36px !important;
    max-height: 36px !important;
}
#new_session_btn:hover {
    background-color: var(--primary-hover) !important;
}

/* Section Titles */
.section-title {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 6px !important;
    margin-top: 4px !important;
}

/* Uploaded Document Cards */
.no-docs-message {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-style: italic;
    padding: 6px 4px;
}
#upload_status {
    font-size: 0.76rem !important;
    color: var(--text-muted) !important;
    margin: -4px 0 0 !important;
}
.uploaded-docs-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 112px;
    overflow-y: auto;
}
#document_list_html,
#upload_status {
    display: none !important;
}
.doc-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 10px;
    display: flex;
    align-items: center;
    gap: 9px;
}
.doc-card:hover {
    border-color: var(--primary);
}
.doc-card-icon {
    color: var(--primary);
    display: flex;
    flex-shrink: 0;
}
.doc-card-content {
    flex-grow: 1;
    overflow: hidden;
}
.doc-card-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-card-subtitle {
    font-size: 0.68rem;
    color: var(--text-muted);
}

/* Recent Session Buttons */
.recent-section {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 0 !important;
    min-height: 260px !important;
    height: 100% !important;
    overflow-y: auto !important;
    gap: 6px !important;
    padding-right: 2px !important;
    margin-top: 2px !important;
}
.recent-section > div {
    width: 100% !important;
}
.no-recent-chats {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    padding: 8px 9px !important;
}

.session-list-btn {
    position: relative !important;
    text-align: left !important;
    justify-content: flex-start !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 8px 34px 8px 9px !important;
    font-size: 0.82rem !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-height: 34px !important;
}
.session-list-btn:hover {
    background-color: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
}
.session-list-btn.primary,
.session-list-btn.primary button,
.session-list-btn button.primary {
    background-color: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
    font-weight: 600 !important;
}

.session-action-trigger {
    position: absolute;
    top: 50%;
    right: 5px;
    z-index: 2;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 5px;
    color: var(--text-muted);
    opacity: 0;
    pointer-events: none;
    transform: translateY(-50%);
    transition: opacity 0.14s ease, background-color 0.14s ease, color 0.14s ease;
}

.session-list-btn:hover .session-action-trigger,
.session-list-btn:focus-within .session-action-trigger,
.session-action-trigger[aria-expanded="true"] {
    opacity: 1;
    pointer-events: auto;
}

.session-action-trigger:hover,
.session-action-trigger:focus-visible {
    background: var(--bg-card);
    color: var(--text-primary);
    outline: none;
}

.session-action-trigger svg,
.session-action-dropdown svg {
    width: 15px;
    height: 15px;
    flex: 0 0 15px;
    stroke-width: 1.8;
}

.session-action-dropdown {
    position: fixed;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    width: 148px;
    padding: 4px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-card);
    box-shadow: var(--shadow-soft);
}

.session-action-item {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    min-height: 30px;
    padding: 5px 7px;
    border: 0;
    border-radius: 4px;
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.78rem;
    font-weight: 500;
    line-height: 1;
    text-align: left;
}

.session-action-item:hover,
.session-action-item:focus-visible {
    background: var(--bg-subtle);
    color: var(--text-primary);
    outline: none;
}

.session-action-item.delete {
    color: var(--danger);
}

.session-action-item.delete:hover,
.session-action-item.delete:focus-visible {
    background: var(--danger-subtle);
}

.session-action-form,
.session-delete-confirm {
    display: flex;
    flex-direction: column;
    gap: 7px;
    padding: 4px;
}

.session-action-form input {
    width: 100%;
    height: 30px;
    padding: 5px 7px;
    border: 1px solid var(--border);
    border-radius: 5px;
    background: var(--bg-input);
    color: var(--text-primary);
    font-size: 0.78rem;
    outline: none;
}

.session-action-form input:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 2px var(--focus-ring);
}

.session-delete-confirm p {
    margin: 0;
    color: var(--text-secondary);
    font-size: 0.76rem;
    line-height: 1.4;
}

.session-action-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 5px;
}

.session-action-compact {
    min-height: 27px;
    padding: 4px 7px;
    border: 0;
    border-radius: 4px;
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.74rem;
    font-weight: 600;
}

.session-action-compact:hover,
.session-action-compact:focus-visible {
    background: var(--bg-subtle);
    color: var(--text-primary);
    outline: none;
}

.session-action-compact.danger {
    color: var(--danger);
}

.session-action-compact.danger:hover,
.session-action-compact.danger:focus-visible {
    background: var(--danger-subtle);
}

.session-action-bridge {
    display: none !important;
}

/* Sidebar footer pinned to bottom */
.sidebar-footer {
    margin-top: auto !important;
    border-top: 1px solid var(--border) !important;
    padding-top: 12px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
}
.sidebar-footer .section-title {
    margin-bottom: 4px !important;
}
#session_box {
    background: transparent !important;
    display: none !important;
}
#session_box input {
    background-color: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    border-radius: 6px !important;
}

#settings_btn {
    width: 100% !important;
    min-height: 34px !important;
    max-height: 34px !important;
    padding: 7px 10px !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: 1px solid var(--border) !important;
}
#settings_btn:hover {
    background: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
}

.settings-drawer {
    position: fixed !important;
    left: 50% !important;
    top: 50% !important;
    bottom: auto !important;
    width: min(460px, calc(100vw - 24px)) !important;
    max-width: calc(100vw - 24px) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-popover) !important;
    padding: 16px !important;
    z-index: 1101 !important;
    color: var(--text-primary) !important;
    transform: translate(-50%, -50%) !important;
}
.settings-drawer,
.settings-drawer * {
    color: var(--text-primary) !important;
}
.settings-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.settings-title {
    font-weight: 700;
    color: var(--text-primary) !important;
}
.settings-drawer label,
.settings-drawer .label-wrap,
.settings-drawer .section-title {
    color: var(--text-secondary) !important;
}
.settings-drawer span,
.settings-drawer p {
    color: var(--text-secondary) !important;
}
.settings-drawer input,
.settings-drawer textarea {
    color: var(--text-primary) !important;
}
.settings-drawer .no-docs-message {
    color: var(--text-muted) !important;
}
.settings-section {
    border-top: 1px solid var(--border);
    padding-top: 10px;
    margin-top: 10px;
}
.settings-doc-row {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 34px !important;
    align-items: center !important;
    gap: 10px !important;
    margin-bottom: 8px !important;
    padding: 9px 10px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.settings-doc-row:hover {
    background: var(--bg-hover) !important;
    border-color: var(--border-hover) !important;
}
.settings-doc-name {
    min-width: 0 !important;
}
.settings-doc-name textarea {
    font-size: 0.82rem !important;
    color: var(--text-primary) !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    min-height: 24px !important;
    height: 24px !important;
    padding: 2px 0 !important;
    resize: none !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.settings-doc-name textarea::placeholder {
    color: var(--text-muted) !important;
}
.doc-delete-btn {
    width: 34px !important;
    min-width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 !important;
    border-radius: 8px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
}
.doc-delete-btn:hover {
    background: var(--danger-subtle) !important;
    border-color: var(--danger-border) !important;
}
.doc-delete-btn:hover,
.doc-delete-btn:hover * {
    color: var(--danger) !important;
}
.delete-confirm-box {
    border-top: 1px solid var(--border);
    margin-top: 8px;
    padding-top: 10px;
}
.delete-confirm-box,
.delete-confirm-box * {
    color: var(--text-primary) !important;
}
.settings-status {
    font-size: 0.8rem !important;
    color: var(--text-primary) !important;
}
.settings-status,
.settings-status * {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}
.settings-drawer button,
.settings-drawer button * {
    color: var(--text-secondary) !important;
}
.settings-drawer button.primary,
.settings-drawer button.stop {
    color: var(--text-on-primary) !important;
}
.settings-drawer button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--border-hover) !important;
}
.settings-drawer button.stop {
    background: var(--danger) !important;
    border-color: var(--danger) !important;
}
.settings-drawer button.stop:hover {
    background: var(--danger-hover) !important;
    border-color: var(--danger-hover) !important;
}
.settings-drawer button.secondary:hover {
    background: var(--bg-hover) !important;
}

/* ── Main Chat Area ───────────────────────────────────────────────── */
.chat-column {
    flex: 1 1 auto !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    background-color: var(--bg-page) !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    overflow: hidden !important;
}

/* Chat Header */
.chat-header {
    background-color: var(--bg-page) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 16px 28px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 16px !important;
    flex-wrap: nowrap !important;
}

.chat-header-leading {
    display: flex !important;
    flex: 1 1 auto !important;
    min-width: 0 !important;
    align-items: center !important;
    gap: 10px !important;
    flex-wrap: nowrap !important;
}

.chat-header-leading > div:last-child {
    min-width: 0 !important;
}
.chat-header-title {
    font-family: 'Manrope', 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
}
.chat-header-subtitle {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* Compact web search control in the composer */
#web_search_toggle {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
    width: auto !important;
    flex: 0 0 auto !important;
    overflow: visible !important;
}
#web_search_toggle .wrap {
    display: flex !important;
    align-items: center !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: visible !important;
}
#web_search_toggle label {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 5px !important;
    width: auto !important;
    min-width: 0 !important;
    height: 32px !important;
    padding: 0 8px !important;
    margin: 0 !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    background: transparent !important;
    color: var(--text-secondary) !important;
    position: relative !important;
    cursor: pointer;
    transition: color 0.15s ease, border-color 0.15s ease, background-color 0.15s ease;
}
#web_search_toggle label:hover {
    background: var(--bg-subtle) !important;
    border-color: var(--border-hover) !important;
    color: var(--text-primary) !important;
}
#web_search_toggle label:focus-visible {
    outline: 2px solid var(--primary) !important;
    outline-offset: 2px !important;
}
#web_search_toggle .label-text {
    font-size: 0.75rem !important;
    line-height: 1 !important;
    font-weight: 500 !important;
    color: inherit !important;
    white-space: nowrap !important;
}
#web_search_toggle .web-search-icon {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 15px !important;
    height: 15px !important;
    color: inherit !important;
    flex: 0 0 15px !important;
}
#web_search_toggle .web-search-icon svg {
    width: 15px !important;
    height: 15px !important;
    stroke: currentColor !important;
}
#web_search_toggle input[type="checkbox"] {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    padding: 0 !important;
    margin: -1px !important;
    overflow: hidden !important;
    clip: rect(0, 0, 0, 0) !important;
    white-space: nowrap !important;
    border: 0 !important;
}
#web_search_toggle.is-active label,
#web_search_toggle label:has(input[type="checkbox"]:checked) {
    background: var(--focus-ring) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}
#web_search_toggle label[data-tooltip]::after {
    content: attr(data-tooltip);
    position: absolute;
    left: 0;
    bottom: calc(100% + 9px);
    z-index: 80;
    width: 230px;
    padding: 7px 9px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-card);
    box-shadow: var(--shadow-popover);
    color: var(--text-primary);
    font-size: 0.72rem;
    font-weight: 400;
    line-height: 1.35;
    text-align: left;
    white-space: normal;
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    transform: translateY(3px);
    transition: opacity 0.12s ease, transform 0.12s ease, visibility 0.12s ease;
}
#web_search_toggle label[data-tooltip]:hover::after,
#web_search_toggle label[data-tooltip]:focus-visible::after {
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
}

/* Chatbot Messages Scrollable Area */
#chatbot {
    flex-grow: 1 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    min-width: 0 !important;
    padding: 24px 28px !important;
    background-color: var(--bg-page) !important;
    border: none !important;
}
#chatbot,
#chatbot > div,
#chatbot .wrap,
#chatbot .bubble-wrap,
#chatbot .empty,
#chatbot .placeholder {
    background: var(--bg-page) !important;
    background-color: var(--bg-page) !important;
    min-width: 0 !important;
    max-width: 100% !important;
}

/* Clean message bubbles */
#chatbot .message-wrap,
#chatbot .bubble-wrap {
    min-width: 0 !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

#chatbot .message {
    padding: 12px 16px !important;
    border-radius: 10px !important;
    line-height: 1.6 !important;
    margin-bottom: 20px !important;
    font-size: 0.92rem !important;
    box-shadow: none !important;
    width: fit-content !important;
    max-width: 85% !important;
    min-width: 0 !important;
    overflow: hidden !important;
    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
    word-break: normal !important;
}

#chatbot .user .message,
#chatbot .bot .message {
    padding: 0 !important;
    margin-bottom: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: auto !important;
    max-width: 100% !important;
    min-width: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

#chatbot .message-buttons:empty {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
}

#chatbot .message-buttons {
    display: none !important;
    min-height: 0 !important;
}

#chatbot button[aria-label="Clear"],
#chatbot button[aria-label="Copy conversation"] {
    display: none !important;
}

#chatbot .user {
    background-color: var(--user-message) !important;
    color: var(--text-on-primary) !important;
    border: none !important;
    align-self: flex-end !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    max-width: 75% !important;
    width: fit-content !important;
    inline-size: fit-content !important;
    min-width: 0 !important;
    overflow: hidden !important;
}
#chatbot .user > *,
#chatbot .message.user > *,
#chatbot [data-testid="user"] .message > *,
#chatbot .bubble-wrap .user > * {
    max-width: 100% !important;
    min-width: 0 !important;
}
#chatbot .user *,
#chatbot .message.user * {
    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
    word-break: normal !important;
    width: auto !important;
    max-width: 100% !important;
}

#chatbot .user.message,
#chatbot .bot.message {
    margin-bottom: 0 !important;
    position: relative !important;
    font-size: 0 !important;
    line-height: 0 !important;
}

#chatbot .user > .message,
#chatbot .user .message-content,
#chatbot .user .md,
#chatbot .user p {
    width: auto !important;
    max-width: 100% !important;
    min-width: 0 !important;
}

#chatbot .bot > .message,
#chatbot .bot .message-content,
#chatbot .bot .md,
#chatbot .bot p {
    width: auto !important;
    max-width: 100% !important;
    min-width: 0 !important;
}

#chatbot .user.message > .message,
#chatbot .bot.message > .message {
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
}

.message-action-trigger {
    position: absolute;
    top: 6px;
    right: 6px;
    z-index: 2;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    padding: 0;
    border: 0;
    border-radius: 5px;
    background: transparent;
    color: var(--text-muted);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.14s ease, background-color 0.14s ease, color 0.14s ease;
}

#chatbot .user.message .message-action-trigger {
    color: rgba(255, 255, 255, 0.72);
}

#chatbot .user.message:hover > .message-action-trigger,
#chatbot .bot.message:hover > .message-action-trigger,
#chatbot .user.message:focus-within > .message-action-trigger,
#chatbot .bot.message:focus-within > .message-action-trigger,
.message-action-trigger[aria-expanded="true"] {
    opacity: 1;
    pointer-events: auto;
}

.message-action-trigger:hover,
.message-action-trigger:focus-visible {
    background: var(--bg-subtle);
    color: var(--text-primary);
    outline: none;
}

#chatbot .user.message .message-action-trigger:hover,
#chatbot .user.message .message-action-trigger:focus-visible {
    background: rgba(255, 255, 255, 0.14);
    color: #FFFFFF;
}

.message-action-trigger svg,
.message-action-dropdown svg {
    width: 15px;
    height: 15px;
    flex: 0 0 15px;
    stroke-width: 1.8;
}

.message-action-dropdown {
    position: fixed;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    width: 116px;
    padding: 4px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-card);
    box-shadow: var(--shadow-soft);
}

.message-action-item {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    min-height: 30px;
    padding: 5px 7px;
    border: 0;
    border-radius: 4px;
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.78rem;
    font-weight: 500;
    line-height: 1;
    text-align: left;
}

.message-action-item:hover,
.message-action-item:focus-visible {
    background: var(--bg-subtle);
    color: var(--text-primary);
    outline: none;
}

.message-action-item.delete {
    color: var(--danger);
}

.message-action-item.delete:hover,
.message-action-item.delete:focus-visible {
    background: var(--danger-subtle);
}

.message-action-bridge {
    display: none !important;
}
#chatbot .bot {
    background-color: var(--assistant-message) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    align-self: flex-start !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    max-width: 85% !important;
    width: fit-content !important;
    inline-size: fit-content !important;
    min-width: 0 !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-soft) !important;
}
#chatbot .message-content,
#chatbot .md,
#chatbot .message p,
#chatbot .message li,
#chatbot .message a {
    min-width: 0 !important;
    max-width: 100% !important;
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}

#chatbot .message-content > .md {
    display: block !important;
    font-size: 0 !important;
    line-height: 0 !important;
}

#chatbot .message-content > .md > * {
    font-size: 1rem;
    line-height: 1.625;
}

#chatbot .message pre {
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    border: 1px solid black !important;
    background: white !important;
    color: black !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    white-space: pre !important;
    overflow-wrap: normal !important;
    word-break: normal !important;
}

#chatbot .message pre code {
    background: transparent !important;
    color: black !important;
    white-space: inherit !important;
    overflow-wrap: normal !important;
    word-break: normal !important;
}
#chatbot .message pre code * {
    color: black !important;
}
.dark #chatbot .message pre,
.dark #chatbot .message pre code {
    background: #000 !important;
    color: #fff !important;
}
.dark #chatbot .message pre code * {
    color: #fff !important;
}
#chatbot .message :not(pre) > code {
    background: transparent !important;
    font-weight: 700 !important;
}
#chatbot .bot,
#chatbot .bot *,
#chatbot .message.bot,
#chatbot .message.bot * {
    color: var(--text-primary) !important;
}

#chatbot [data-testid="status-tracker"],
#chatbot [data-testid="status-tracker"].wrap,
#chatbot .message-wrap:has(.message.bot.pending) {
    height: auto !important;
    min-height: 0 !important;
    flex: 0 0 auto !important;
    justify-content: flex-start !important;
}

#chatbot [data-testid="status-tracker"] {
    width: fit-content !important;
    background: transparent !important;
    background-color: transparent !important;
}

#chatbot .message.bot.pending,
#chatbot .message.bot.pending .message-content {
    height: auto !important;
    min-height: 0 !important;
}

#chatbot .message.bot.pending .message-content {
    padding: 0 !important;
}

/* Sources block rendered inside an assistant message */
.source-note {
    margin-top: 10px;
    padding: 7px 10px;
    border-left: 2px solid var(--primary);
    background: var(--bg-subtle);
    color: var(--text-secondary);
    font-size: 0.78rem;
    border-radius: 4px;
}
details.sources-panel {
    margin-top: 10px;
}
details.sources-panel summary {
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-secondary);
    list-style: none;
}
details.sources-panel summary::-webkit-details-marker {
    display: none;
}
details.sources-panel summary::before {
    content: "›";
    display: inline-block;
    margin-right: 6px;
    transition: transform 0.15s ease;
}
details.sources-panel[open] summary::before {
    transform: rotate(90deg);
}
.source-item {
    display: flex;
    gap: 10px;
    padding: 10px 0;
    border-top: 1px solid var(--border);
}
.source-index {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--primary);
    font-weight: 700;
    flex-shrink: 0;
    padding-top: 2px;
}
.source-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-primary);
}
.source-link {
    font-size: 0.74rem;
    color: var(--link);
    word-break: break-all;
}
.source-snippet {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 3px;
    line-height: 1.45;
}

/* Welcome Centerpiece */
.welcome-container {
    text-align: center;
    margin: auto !important;
    max-width: 440px;
    padding: 40px 20px;
    background: var(--bg-page) !important;
}
.welcome-mark {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    color: var(--text-muted);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 16px;
}
.welcome-title {
    font-family: 'Manrope', 'Inter', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 10px;
    letter-spacing: -0.01em;
}
.welcome-subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.55;
}

/* ── Fixed Bottom Input Area ────────────────────────────────────────── */
.input-area-wrapper {
    padding: 14px 28px 22px !important;
    background-color: var(--bg-page) !important;
    border-top: 1px solid var(--border) !important;
}

.progress-level,
.wrap .progress-level,
[class*="progress"],
[class*="toast"] {
    pointer-events: none !important;
}

.input-area-wrapper [class*="progress"],
.input-container-row [class*="progress"] {
    position: static !important;
    inset: auto !important;
    max-height: 18px !important;
    font-size: 0.72rem !important;
}

.input-container-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 6px 8px !important;
    align-items: center !important;
    height: 50px !important;
    gap: 8px !important;
    max-width: 760px;
    margin: auto;
}

.chat-upload,
#upload_btn {
    width: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    flex: 0 0 34px !important;
    overflow: hidden !important;
}
#upload_btn {
    border-radius: 50% !important;
    color: var(--text-secondary) !important;
    font-size: 22px !important;
    font-weight: 400 !important;
    line-height: 1 !important;
}
#upload_btn:hover {
    background: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
}
#upload_btn button {
    width: 34px !important;
    min-width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#upload_btn button::before {
    content: "+" !important;
    color: var(--text-secondary) !important;
    font-size: 22px !important;
    font-weight: 400 !important;
    line-height: 1 !important;
}
.chat-upload .wrap,
.chat-upload .label-wrap {
    width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.chat-upload .wrap {
    border-radius: 50% !important;
    color: transparent !important;
    cursor: pointer !important;
    position: relative !important;
}
.chat-upload .wrap:hover {
    background: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
}
.chat-upload span,
.chat-upload p,
.chat-upload .or,
.chat-upload .file-preview,
.chat-upload .upload-text {
    display: none !important;
}
.chat-upload svg {
    display: none !important;
}
.chat-upload .wrap::before {
    content: "📎";
    font-size: 17px;
    line-height: 1;
}

.input-container-row:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--focus-ring) !important;
}

.chat-upload .wrap::before,
.chat-upload .label-wrap::before {
    content: "+" !important;
    color: var(--text-secondary) !important;
    font-size: 22px !important;
    font-weight: 400 !important;
    line-height: 1 !important;
}

#msg_input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 0 !important;
    min-width: 0 !important;
    flex: 1 1 0 !important;
}

#msg_input textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-primary) !important;
    font-size: 0.92rem !important;
    padding: 8px 4px !important;
}

/* Rounded Send Button */
#send_btn {
    background-color: var(--primary) !important;
    color: var(--text-on-primary) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
    flex-shrink: 0 !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
    font-weight: bold !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#send_btn:hover {
    background-color: var(--primary-hover) !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--border-hover);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

footer {
    display: none !important;
}
"""

PROCESSING_INDICATOR_JS = """
() => {
    const indicatorClasses = ["bg-transparent", "text-black"];
    const visualClassPrefixes = ["opacity-", "shadow", "border", "backdrop"];
    const visualClasses = new Set(["bg-slate-900"]);
    const selector = [
        ".input-area-wrapper [class*='progress']",
        ".input-container-row [class*='progress']",
        ".input-area-wrapper [class*='toast']",
        ".input-container-row [class*='toast']"
    ].join(",");

    const scrubIndicator = (element) => {
        [...element.classList].forEach((className) => {
            if (
                visualClasses.has(className) ||
                visualClassPrefixes.some((prefix) => className.startsWith(prefix))
            ) {
                element.classList.remove(className);
            }
        });

        element.classList.add(...indicatorClasses);
    };

    let scheduled = false;
    const applyIndicatorTheme = () => {
        scheduled = false;
        document.querySelectorAll(selector).forEach(scrubIndicator);
    };
    const scheduleIndicatorTheme = () => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(applyIndicatorTheme);
    };

    scheduleIndicatorTheme();

    if (!window.processingIndicatorThemeObserver) {
        window.processingIndicatorThemeObserver = new MutationObserver(scheduleIndicatorTheme);
        window.processingIndicatorThemeObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
}
"""

MESSAGE_ACTIONS_JS = """
() => {
    const icon = (name) => {
        const paths = {
            more: '<circle cx="5" cy="12" r="1"></circle><circle cx="12" cy="12" r="1"></circle><circle cx="19" cy="12" r="1"></circle>',
            copy: '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path>',
            trash: '<path d="M3 6h18"></path><path d="M8 6V4c0-1 .8-2 2-2h4c1.2 0 2 1 2 2v2"></path><path d="M19 6l-1 14c-.1 1.1-1 2-2 2H8c-1 0-1.9-.9-2-2L5 6"></path><path d="M10 11v6"></path><path d="M14 11v6"></path>'
        };
        return `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name]}</svg>`;
    };

    const closeMenu = () => {
        document.querySelectorAll(".message-action-dropdown").forEach((menu) => menu.remove());
        document.querySelectorAll(".message-action-trigger").forEach((button) => {
            button.setAttribute("aria-expanded", "false");
        });
    };

    const setDeleteIndex = (index) => {
        const input = document.querySelector("#message_delete_index input");
        const triggerRoot = document.querySelector("#message_delete_trigger");
        const trigger = triggerRoot?.matches("button")
            ? triggerRoot
            : triggerRoot?.querySelector("button");
        if (!input || !trigger) return;

        const valueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype,
            "value"
        ).set;
        valueSetter.call(input, String(index));
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
        trigger.click();
    };

    const addMenu = (row, index) => {
        const bubble = row.querySelector(":scope .message.user, :scope .message.bot");
        if (!bubble || bubble.querySelector(":scope > .message-action-trigger")) return;

        const nativeControls = row.nextElementSibling?.classList.contains("message-buttons")
            ? row.nextElementSibling
            : null;
        const nativeCopy = nativeControls?.querySelector('button[aria-label="Copy message"]');

        const trigger = document.createElement("button");
        trigger.type = "button";
        trigger.className = "message-action-trigger";
        trigger.setAttribute("aria-label", "Message actions");
        trigger.setAttribute("aria-haspopup", "menu");
        trigger.setAttribute("aria-expanded", "false");
        trigger.innerHTML = icon("more");

        trigger.addEventListener("click", (event) => {
            event.stopPropagation();
            const wasOpen = trigger.getAttribute("aria-expanded") === "true";
            closeMenu();
            if (wasOpen) return;

            const menu = document.createElement("div");
            menu.className = "message-action-dropdown";
            menu.setAttribute("role", "menu");
            menu.innerHTML = `
                <button type="button" class="message-action-item copy" role="menuitem">
                    ${icon("copy")}<span>Copy</span>
                </button>
                <button type="button" class="message-action-item delete" role="menuitem">
                    ${icon("trash")}<span>Delete</span>
                </button>
            `;
            document.body.appendChild(menu);

            const rect = trigger.getBoundingClientRect();
            const menuWidth = 116;
            menu.style.top = `${Math.min(rect.bottom + 4, window.innerHeight - 76)}px`;
            menu.style.left = `${Math.max(8, Math.min(rect.right - menuWidth, window.innerWidth - menuWidth - 8))}px`;
            trigger.setAttribute("aria-expanded", "true");

            menu.querySelector(".copy").addEventListener("click", () => {
                const messageText = bubble.querySelector(".message-content")?.innerText || "";
                if (navigator.clipboard?.writeText) {
                    navigator.clipboard.writeText(messageText).catch(() => nativeCopy?.click());
                } else {
                    nativeCopy?.click();
                }
                closeMenu();
            });
            menu.querySelector(".delete").addEventListener("click", () => {
                closeMenu();
                setDeleteIndex(index);
            });
        });

        bubble.appendChild(trigger);
    };

    let scheduled = false;
    const enhanceMessages = () => {
        scheduled = false;
        const rows = [...document.querySelectorAll("#chatbot .message-row")];
        rows.forEach(addMenu);
    };
    const scheduleEnhancement = () => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(enhanceMessages);
    };

    if (!window.messageActionsInitialized) {
        window.messageActionsInitialized = true;
        document.addEventListener("click", (event) => {
            if (!event.target.closest(".message-action-trigger, .message-action-dropdown")) {
                closeMenu();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") closeMenu();
        });
        window.addEventListener("resize", closeMenu);
        window.addEventListener("scroll", closeMenu, true);
        window.messageActionsObserver = new MutationObserver(scheduleEnhancement);
        window.messageActionsObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    scheduleEnhancement();
}
"""

RECENT_CHAT_ACTIONS_JS = """
() => {
    const icon = (name) => {
        const paths = {
            more: '<circle cx="5" cy="12" r="1"></circle><circle cx="12" cy="12" r="1"></circle><circle cx="19" cy="12" r="1"></circle>',
            pencil: '<path d="M21.2 6.8a2.4 2.4 0 0 0-3.4-3.4L5 16.2 3.8 21l4.8-1.2Z"></path><path d="m15.5 5.7 3.4 3.4"></path>',
            trash: '<path d="M3 6h18"></path><path d="M8 6V4c0-1 .8-2 2-2h4c1.2 0 2 1 2 2v2"></path><path d="M19 6l-1 14c-.1 1.1-1 2-2 2H8c-1 0-1.9-.9-2-2L5 6"></path><path d="M10 11v6"></path><path d="M14 11v6"></path>'
        };
        return `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name]}</svg>`;
    };

    const closeMenu = () => {
        document.querySelectorAll(".session-action-dropdown").forEach((menu) => menu.remove());
        document.querySelectorAll(".session-action-trigger").forEach((trigger) => {
            trigger.setAttribute("aria-expanded", "false");
        });
    };

    const componentControl = (selector, controlSelector) => {
        const root = document.querySelector(selector);
        if (!root) return null;
        return root.matches(controlSelector) ? root : root.querySelector(controlSelector);
    };

    const setFieldValue = (selector, value) => {
        const field = componentControl(selector, "input, textarea");
        if (!field) return false;
        const prototype = field instanceof window.HTMLTextAreaElement
            ? window.HTMLTextAreaElement.prototype
            : window.HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(prototype, "value").set;
        setter.call(field, String(value));
        field.dispatchEvent(new Event("input", { bubbles: true }));
        field.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
    };

    const runAction = (action, index, title = "") => {
        const indexSet = setFieldValue("#conversation_action_index", index);
        const titleSet = action !== "rename"
            || setFieldValue("#conversation_rename_title", title);
        const trigger = componentControl(
            action === "rename"
                ? "#conversation_rename_trigger"
                : "#conversation_delete_trigger",
            "button"
        );
        if (indexSet && titleSet && trigger) trigger.click();
    };

    const positionMenu = (menu, trigger) => {
        const rect = trigger.getBoundingClientRect();
        const menuWidth = 148;
        menu.style.top = `${Math.min(rect.bottom + 4, window.innerHeight - menu.offsetHeight - 8)}px`;
        menu.style.left = `${Math.max(8, Math.min(rect.right - menuWidth, window.innerWidth - menuWidth - 8))}px`;
    };

    const showRename = (menu, trigger, index, currentTitle) => {
        menu.innerHTML = `
            <div class="session-action-form">
                <input type="text" maxlength="80" aria-label="Conversation name">
                <div class="session-action-buttons">
                    <button type="button" class="session-action-compact cancel">Cancel</button>
                    <button type="button" class="session-action-compact save">Save</button>
                </div>
            </div>
        `;
        const input = menu.querySelector("input");
        input.value = currentTitle;
        input.focus({ preventScroll: true });
        input.select();
        positionMenu(menu, trigger);

        const save = () => {
            const title = input.value.trim();
            if (!title) return;
            runAction("rename", index, title);
            closeMenu();
        };
        menu.querySelector(".save").addEventListener("click", save);
        menu.querySelector(".cancel").addEventListener("click", closeMenu);
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                save();
            }
            if (event.key === "Escape") closeMenu();
        });
    };

    const showDeleteConfirmation = (menu, trigger, index) => {
        menu.innerHTML = `
            <div class="session-delete-confirm">
                <p>Delete this conversation? This cannot be undone.</p>
                <div class="session-action-buttons">
                    <button type="button" class="session-action-compact cancel">Cancel</button>
                    <button type="button" class="session-action-compact danger confirm">Delete</button>
                </div>
            </div>
        `;
        positionMenu(menu, trigger);
        menu.querySelector(".cancel").addEventListener("click", closeMenu);
        menu.querySelector(".confirm").addEventListener("click", () => {
            runAction("delete", index);
            closeMenu();
        });
    };

    const openMenu = (event, trigger, row, index) => {
        event.preventDefault();
        event.stopPropagation();
        const wasOpen = trigger.getAttribute("aria-expanded") === "true";
        closeMenu();
        if (wasOpen) return;

        const menu = document.createElement("div");
        menu.className = "session-action-dropdown";
        menu.setAttribute("role", "menu");
        menu.addEventListener("click", (menuEvent) => menuEvent.stopPropagation());
        menu.innerHTML = `
            <button type="button" class="session-action-item rename" role="menuitem">
                ${icon("pencil")}<span>Rename</span>
            </button>
            <button type="button" class="session-action-item delete" role="menuitem">
                ${icon("trash")}<span>Delete</span>
            </button>
        `;
        document.body.appendChild(menu);
        trigger.setAttribute("aria-expanded", "true");
        positionMenu(menu, trigger);

        const currentTitle = [...row.childNodes]
            .filter((node) => node !== trigger)
            .map((node) => node.textContent || "")
            .join("")
            .trim();
        menu.querySelector(".rename").addEventListener("click", () => {
            showRename(menu, trigger, index, currentTitle);
        });
        menu.querySelector(".delete").addEventListener("click", () => {
            showDeleteConfirmation(menu, trigger, index);
        });
    };

    const enhanceRows = () => {
        document.querySelectorAll(".session-list-btn").forEach((row, index) => {
            const existingTrigger = row.querySelector(":scope > .session-action-trigger");
            const rowTitle = [...row.childNodes]
                .filter((node) => node !== existingTrigger)
                .map((node) => node.textContent || "")
                .join("")
                .trim();
            if (rowTitle) row.setAttribute("aria-label", rowTitle);
            if (existingTrigger) return;
            const trigger = document.createElement("span");
            trigger.className = "session-action-trigger";
            trigger.setAttribute("role", "button");
            trigger.setAttribute("tabindex", "0");
            trigger.setAttribute("aria-label", "Conversation actions");
            trigger.setAttribute("aria-haspopup", "menu");
            trigger.setAttribute("aria-expanded", "false");
            trigger.innerHTML = icon("more");
            trigger.addEventListener("pointerdown", (event) => {
                event.preventDefault();
                event.stopPropagation();
            });
            trigger.addEventListener("click", (event) => openMenu(event, trigger, row, index));
            trigger.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    openMenu(event, trigger, row, index);
                }
            });
            row.appendChild(trigger);
        });
    };

    let scheduled = false;
    const scheduleEnhancement = () => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(() => {
            scheduled = false;
            enhanceRows();
        });
    };

    if (!window.recentChatActionsInitialized) {
        window.recentChatActionsInitialized = true;
        document.addEventListener("click", (event) => {
            if (!event.target.closest(".session-action-trigger, .session-action-dropdown")) {
                closeMenu();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") closeMenu();
        });
        window.addEventListener("resize", closeMenu);
        document.querySelector(".recent-section")?.addEventListener("scroll", () => {
            if (!document.querySelector(".session-action-form input:focus")) {
                closeMenu();
            }
        });
        window.recentChatActionsObserver = new MutationObserver(scheduleEnhancement);
        window.recentChatActionsObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    scheduleEnhancement();
}
"""

SIDEBAR_DRAWER_JS = """
() => {
    const icon = (name) => {
        const paths = {
            menu: '<path d="M4 6h16"></path><path d="M4 12h16"></path><path d="M4 18h16"></path>',
            close: '<path d="M18 6 6 18"></path><path d="m6 6 12 12"></path>',
            collapse: '<rect width="18" height="18" x="3" y="3" rx="2"></rect><path d="M9 3v18"></path><path d="m16 15-3-3 3-3"></path>'
        };
        return `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name]}</svg>`;
    };

    const buttonControl = (selector) => {
        const root = document.querySelector(selector);
        if (!root) return null;
        return root.matches("button") ? root : root.querySelector("button");
    };

    const drawer = document.querySelector(".sidebar-column");
    const openButton = buttonControl("#sidebar_open_btn");
    const closeButton = buttonControl("#sidebar_close_btn");
    if (!drawer || !openButton || !closeButton) return;

    drawer.id = drawer.id || "recent_chats_drawer";
    openButton.innerHTML = icon("menu");
    openButton.setAttribute("aria-label", "Open recent chats");
    openButton.setAttribute("aria-controls", drawer.id);
    openButton.setAttribute("aria-expanded", "false");

    let backdrop = document.querySelector(".sidebar-drawer-backdrop");
    if (!backdrop) {
        backdrop = document.createElement("button");
        backdrop.type = "button";
        backdrop.className = "sidebar-drawer-backdrop";
        backdrop.setAttribute("aria-label", "Close recent chats");
        document.body.appendChild(backdrop);
    }

    const desktopQuery = window.matchMedia("(min-width: 768px)");
    window.isSidebarOpen = false;
    window.isSidebarCollapsed = false;

    const renderSidebarState = () => {
        const isDesktop = desktopQuery.matches;
        const mobileOpen = !isDesktop && window.isSidebarOpen;
        const desktopCollapsed = isDesktop && window.isSidebarCollapsed;

        if (isDesktop) {
            backdrop.style.setProperty("display", "none", "important");
        } else {
            backdrop.style.removeProperty("display");
        }
        document.body.classList.toggle("sidebar-open", mobileOpen);
        document.documentElement.classList.toggle("sidebar-open", mobileOpen);
        drawer.classList.toggle("is-open", mobileOpen);
        drawer.classList.toggle("desktop-collapsed", desktopCollapsed);
        drawer.setAttribute("aria-hidden", String(!isDesktop && !mobileOpen));
        openButton.setAttribute("aria-expanded", String(mobileOpen));

        if (isDesktop) {
            closeButton.innerHTML = icon(desktopCollapsed ? "menu" : "collapse");
            closeButton.setAttribute(
                "aria-label",
                desktopCollapsed ? "Expand recent chats" : "Collapse recent chats"
            );
            closeButton.setAttribute("aria-expanded", String(!desktopCollapsed));
        } else {
            closeButton.innerHTML = icon("close");
            closeButton.setAttribute("aria-label", "Close recent chats");
            closeButton.removeAttribute("aria-expanded");
        }
    };

    const setSidebarOpen = (isOpen, restoreFocus = false) => {
        if (desktopQuery.matches) return;
        window.isSidebarOpen = Boolean(isOpen);
        renderSidebarState();

        if (!window.isSidebarOpen) {
            document.querySelectorAll(
                ".session-action-dropdown, .message-action-dropdown"
            ).forEach((menu) => menu.remove());
        }

        requestAnimationFrame(() => {
            if (window.isSidebarOpen) {
                closeButton.focus({ preventScroll: true });
            } else if (restoreFocus) {
                openButton.focus({ preventScroll: true });
            }
        });
    };

    const setSidebarCollapsed = (isCollapsed) => {
        if (!desktopQuery.matches) return;
        window.isSidebarCollapsed = Boolean(isCollapsed);
        renderSidebarState();
        requestAnimationFrame(() => {
            closeButton.focus({ preventScroll: true });
        });
    };

    window.setSidebarOpen = setSidebarOpen;
    window.setSidebarCollapsed = setSidebarCollapsed;
    renderSidebarState();

    if (!window.sidebarDrawerInitialized) {
        window.sidebarDrawerInitialized = true;
        openButton.addEventListener("click", (event) => {
            event.preventDefault();
            setSidebarOpen(true);
        });
        closeButton.addEventListener("click", (event) => {
            event.preventDefault();
            if (desktopQuery.matches) {
                setSidebarCollapsed(!window.isSidebarCollapsed);
            } else {
                setSidebarOpen(false, true);
            }
        });
        backdrop.addEventListener("click", () => setSidebarOpen(false, true));
        document.addEventListener("keydown", (event) => {
            if (
                event.key === "Escape"
                && !desktopQuery.matches
                && window.isSidebarOpen
            ) {
                event.preventDefault();
                setSidebarOpen(false, true);
            }
        });
        document.addEventListener("click", (event) => {
            const conversation = event.target.closest(".session-list-btn");
            if (
                conversation
                && !event.target.closest(".session-action-trigger")
                && !desktopQuery.matches
            ) {
                setSidebarOpen(false);
            }
        });
        document.querySelector("#new_session_btn")?.addEventListener(
            "click",
            () => {
                if (!desktopQuery.matches) setSidebarOpen(false);
            }
        );
        desktopQuery.addEventListener("change", () => {
            window.isSidebarOpen = false;
            renderSidebarState();
        });
    }
}
"""

WEB_SEARCH_TOGGLE_JS = """
() => {
    const tooltip = "Search the web when your documents don’t contain the answer.";
    const globe = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
             aria-hidden="true">
            <circle cx="12" cy="12" r="10"></circle>
            <path d="M2 12h20"></path>
            <path d="M12 2a15.3 15.3 0 0 1 0 20"></path>
            <path d="M12 2a15.3 15.3 0 0 0 0 20"></path>
        </svg>`;

    const enhanceToggle = () => {
        const root = document.querySelector("#web_search_toggle");
        const input = root?.querySelector('input[type="checkbox"]');
        const label = root?.querySelector("label");
        const labelText = label?.querySelector(".label-text");
        if (!root || !input || !label || !labelText) return;

        if (!label.querySelector(".web-search-icon")) {
            const icon = document.createElement("span");
            icon.className = "web-search-icon";
            icon.innerHTML = globe;
            label.insertBefore(icon, labelText);
        }

        label.setAttribute("role", "button");
        label.setAttribute("tabindex", "0");
        label.setAttribute("data-tooltip", tooltip);
        label.setAttribute("title", tooltip);
        input.setAttribute("tabindex", "-1");

        const sync = () => {
            const enabled = Boolean(input.checked);
            window.isWebSearchEnabled = enabled;
            label.setAttribute("aria-pressed", String(enabled));
            root.classList.toggle("is-active", enabled);
        };

        if (!input.dataset.webSearchBound) {
            input.dataset.webSearchBound = "true";
            input.addEventListener("change", sync);
        }
        if (!label.dataset.webSearchKeyboardBound) {
            label.dataset.webSearchKeyboardBound = "true";
            label.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    input.click();
                }
            });
        }
        sync();
    };

    let frame = 0;
    const scheduleEnhance = () => {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(enhanceToggle);
    };
    enhanceToggle();
    new MutationObserver(scheduleEnhance).observe(document.body, {
        childList: true,
        subtree: true
    });
}
"""

CHAT_SUBMIT_START_JS = """
(query, chatHistory, sessionId, webSearch) => {
    if (window.chatSubmitAbortController) {
        window.chatSubmitAbortController.abort();
    }
    window.chatSubmitAbortController = new AbortController();
    window.lastChatSubmitRequest = {
        sessionId,
        webSearch: Boolean(webSearch)
    };
    return [query, chatHistory, sessionId, webSearch];
}
"""


with gr.Blocks() as demo:

    # ── State ────────────────────────────────────────────────────────
    session_state = gr.State(None)
    recent_sessions_state = gr.State([])
    web_search_session_state = gr.State({})

    # ── Main Two-Column Row ──────────────────────────────────────────
    with gr.Row(elem_classes="main-row"):

        # ── Left Sidebar ───────────────────────────────────────────────
        with gr.Column(elem_classes="sidebar-column"):

            # App Logo and Title
            with gr.Row(elem_classes="drawer-header"):
                gr.HTML(
                    "<div class='logo-container'>"
                    f"{LOGO_MARK_HTML}"
                    "<span class='logo-title'>Document Assistant</span>"
                    "</div>"
                )
                sidebar_close_btn = gr.Button(
                    "Close",
                    variant="secondary",
                    elem_id="sidebar_close_btn",
                    scale=0,
                )

            # New Session Button
            new_session_btn = gr.Button(
                "+ New chat",
                variant="primary",
                elem_id="new_session_btn",
            )

            document_list_html = gr.HTML(
                value=get_uploaded_documents_html(),
                elem_id="document_list_html",
                visible=False,
            )
            upload_status = gr.Markdown(
                value="*Upload a PDF to get started.*",
                elem_id="upload_status",
                visible=False,
            )

            # Recent Sessions Section
            with gr.Column(elem_classes="recent-section"):
                gr.HTML("<div class='section-title'>Recent Chats</div>")
                no_recent_chats = gr.HTML(
                    "<div class='no-recent-chats'>No recent chats</div>",
                    visible=False,
                )

                session_btns = []
                for i in range(RECENT_SESSION_LIMIT):
                    btn = gr.Button(
                        visible=False,
                        variant="secondary",
                        elem_classes="session-list-btn",
                    )
                    session_btns.append(btn)

            # Session footer pinned to the bottom of the sidebar
            with gr.Column(elem_classes="sidebar-footer"):
                settings_btn = gr.Button(
                    "Settings",
                    variant="secondary",
                    elem_id="settings_btn",
                )
                session_box = gr.Textbox(
                    show_label=False,
                    container=False,
                    interactive=False,
                    elem_id="session_box",
                )

        # ── Main Chat Area (Flexible) ────────────────────────────────
        with gr.Column(elem_classes="chat-column"):

            # Chat Header
            with gr.Row(elem_classes="chat-header"):
                with gr.Row(elem_classes="chat-header-leading"):
                    sidebar_open_btn = gr.Button(
                        "Menu",
                        variant="secondary",
                        elem_id="sidebar_open_btn",
                        scale=0,
                    )
                    gr.HTML(
                        "<div>"
                        "<div class='chat-header-title'>Document Assistant</div>"
                        "<div class='chat-header-subtitle'>"
                        "Ask questions about your documents. Turn on web search to fall back "
                        "to the web when the answer isn't in your documents."
                        "</div></div>"
                    )
            # Chatbot Container
            chatbot = gr.Chatbot(
                label=None,
                show_label=False,
                elem_id="chatbot",
                placeholder=(
                    "<div class='welcome-container'>"
                    "<div class='welcome-title'>How can I help you today?</div>"
                    "</div>"
                ),
                render_markdown=True,
                layout="bubble",
                buttons=["copy"],
            )
            message_delete_index = gr.Number(
                value=-1,
                precision=0,
                show_label=False,
                container=False,
                elem_id="message_delete_index",
                elem_classes="message-action-bridge",
            )
            message_delete_trigger = gr.Button(
                "Delete message",
                elem_id="message_delete_trigger",
                elem_classes="message-action-bridge",
            )
            conversation_action_index = gr.Number(
                value=-1,
                precision=0,
                show_label=False,
                container=False,
                elem_id="conversation_action_index",
                elem_classes="session-action-bridge",
            )
            conversation_rename_title = gr.Textbox(
                value="",
                show_label=False,
                container=False,
                elem_id="conversation_rename_title",
                elem_classes="session-action-bridge",
            )
            conversation_rename_trigger = gr.Button(
                "Rename conversation",
                elem_id="conversation_rename_trigger",
                elem_classes="session-action-bridge",
            )
            conversation_delete_trigger = gr.Button(
                "Delete conversation",
                elem_id="conversation_delete_trigger",
                elem_classes="session-action-bridge",
            )

            # Fixed Bottom Input Area
            with gr.Column(elem_classes="input-area-wrapper"):
                with gr.Row(elem_classes="input-container-row"):
                    file_input = gr.UploadButton(
                        "+",
                        variant="secondary",
                        size="sm",
                        file_types=[".pdf"],
                        type="filepath",
                        elem_id="upload_btn",
                        elem_classes="chat-upload",
                        scale=0,
                        min_width=34,
                    )
                    web_search_toggle = gr.Checkbox(
                        label="Web search",
                        value=False,
                        elem_id="web_search_toggle",
                        container=False,
                    )
                    msg_input = gr.Textbox(
                        placeholder="Ask anything...",
                        show_label=False,
                        container=False,
                        autofocus=True,
                        elem_id="msg_input"
                    )
                    send_btn = gr.Button("↑", variant="primary", elem_id="send_btn", scale=0)

    # ── Event wiring ─────────────────────────────────────────────────

    selected_delete_doc = gr.State(None)
    pending_query = gr.State("")
    pending_request_id = gr.State("")

    with gr.Column(visible=False, elem_classes="settings-drawer") as settings_panel:
        with gr.Row(elem_classes="settings-header"):
            gr.HTML("<div class='settings-title'>Settings</div>")
            close_settings_btn = gr.Button("×", variant="secondary", elem_classes="doc-delete-btn", scale=0)

        appearance_toggle = gr.Radio(
            choices=["Light", "Dark"],
            value="Light",
            label="Appearance",
            elem_classes="settings-section",
        )

        gr.HTML("<div class='settings-section'><div class='section-title'>Uploaded Documents</div></div>")
        no_uploaded_docs = gr.HTML(
            "<div class='no-docs-message'>No documents uploaded yet.</div>",
            visible=False,
        )

        settings_doc_rows = []
        settings_doc_names = []
        settings_delete_btns = []
        for i in range(SETTINGS_DOC_LIMIT):
            with gr.Row(visible=False, elem_classes="settings-doc-row") as doc_row:
                doc_name = gr.Textbox(
                    value="",
                    show_label=False,
                    container=False,
                    interactive=False,
                    elem_classes="settings-doc-name",
                )
                delete_btn = gr.Button("🗑", variant="secondary", elem_classes="doc-delete-btn", scale=0)
                settings_doc_rows.append(doc_row)
                settings_doc_names.append(doc_name)
                settings_delete_btns.append(delete_btn)

        with gr.Column(visible=False, elem_classes="delete-confirm-box") as delete_confirm_box:
            delete_confirm_text = gr.Markdown("Delete this PDF?")
            with gr.Row():
                confirm_delete_btn = gr.Button("Delete", variant="stop", size="sm")
                cancel_delete_btn = gr.Button("Cancel", variant="secondary", size="sm")

        settings_status = gr.Markdown("", elem_classes="settings-status")
        settings_doc_outputs = [no_uploaded_docs] + [
            component
            for row in zip(settings_doc_rows, settings_doc_names, settings_delete_btns)
            for component in row
        ]

    # Show session id on load and populate session state, document list, and recent sessions list
    demo.load(fn=None, js=PROCESSING_INDICATOR_JS)
    demo.load(fn=None, js=MESSAGE_ACTIONS_JS)
    demo.load(fn=None, js=RECENT_CHAT_ACTIONS_JS)
    demo.load(fn=None, js=SIDEBAR_DRAWER_JS)
    demo.load(fn=None, js=WEB_SEARCH_TOGGLE_JS)

    demo.load(
        fn=on_app_load,
        inputs=session_state,
        outputs=[recent_sessions_state, document_list_html, no_recent_chats] + session_btns + [session_box,session_state],
    )

    settings_btn.click(
        fn=open_settings_panel,
        outputs=[
            settings_panel,
            settings_status,
            delete_confirm_box,
            selected_delete_doc,
            delete_confirm_text,
        ] + settings_doc_outputs,
    )

    close_settings_btn.click(
        fn=close_settings_panel,
        outputs=[
            settings_panel,
            delete_confirm_box,
            selected_delete_doc,
            settings_status,
            delete_confirm_text,
        ],
    )

    appearance_toggle.change(
        fn=None,
        inputs=appearance_toggle,
        js="(mode) => { document.body.classList.toggle('dark-mode', mode === 'Dark'); return mode; }",
    )

    # Upload (Triggered automatically when file changes)
    file_input.upload(
        fn=upload_pdf,
        inputs=file_input,
        outputs=[upload_status, document_list_html],
        show_progress="hidden",
    ).then(
        fn=lambda: _settings_document_updates(),
        outputs=settings_doc_outputs,
    )

    for i, delete_btn in enumerate(settings_delete_btns):
        delete_btn.click(
            fn=lambda index=i: prepare_delete_document(index),
            outputs=[delete_confirm_box, selected_delete_doc, settings_status, delete_confirm_text],
        )

    cancel_delete_btn.click(
        fn=lambda: (gr.update(visible=False), None, "", "Delete this PDF?"),
        outputs=[delete_confirm_box, selected_delete_doc, settings_status, delete_confirm_text],
    )

    confirm_delete_btn.click(
        fn=delete_document,
        inputs=selected_delete_doc,
        outputs=[
            delete_confirm_box,
            selected_delete_doc,
            settings_status,
            document_list_html,
        ] + settings_doc_outputs,
    )

    message_delete_trigger.click(
        fn=delete_chat_message,
        inputs=[chatbot, message_delete_index],
        outputs=chatbot,
        queue=False,
        show_progress="hidden",
    )

    web_search_toggle.change(
        fn=remember_web_search_setting,
        inputs=[web_search_toggle, session_state, web_search_session_state],
        outputs=web_search_session_state,
        queue=False,
        show_progress="hidden",
    )

    conversation_rename_trigger.click(
        fn=rename_conversation,
        inputs=[
            conversation_action_index,
            conversation_rename_title,
            recent_sessions_state,
            session_state,
        ],
        outputs=[recent_sessions_state, no_recent_chats] + session_btns,
        queue=False,
        show_progress="hidden",
    )

    conversation_delete_trigger.click(
        fn=delete_conversation,
        inputs=[
            conversation_action_index,
            recent_sessions_state,
            session_state,
            web_search_session_state,
        ],
        outputs=[
            recent_sessions_state,
            session_state,
            session_box,
            chatbot,
            web_search_toggle,
            web_search_session_state,
            no_recent_chats,
        ] + session_btns,
        queue=False,
        show_progress="hidden",
    )

    # Send message (button or Enter) and then refresh recent sessions
    send_click = send_btn.click(
        fn=begin_ask,
        inputs=[msg_input, chatbot, session_state, web_search_toggle],
        outputs=[chatbot, msg_input, pending_query, pending_request_id],
        queue=False,
        show_progress="hidden",
        js=CHAT_SUBMIT_START_JS,
    )
    send_response = send_click.then(
        fn=ask,
        inputs=[pending_query, chatbot, session_state, web_search_toggle, pending_request_id],
        outputs=chatbot,
        show_progress="minimal",
        trigger_mode="always_last",
        concurrency_limit=1,
        concurrency_id="chat_send",
    )
    send_response.then(
        fn=refresh_sessions_after_send,
        inputs=[session_state, recent_sessions_state],
        outputs=[recent_sessions_state, no_recent_chats] + session_btns
    )

    submit_start = msg_input.submit(
        fn=begin_ask,
        inputs=[msg_input, chatbot, session_state, web_search_toggle],
        outputs=[chatbot, msg_input, pending_query, pending_request_id],
        queue=False,
        show_progress="hidden",
        js=CHAT_SUBMIT_START_JS,
    )
    submit_response = submit_start.then(
        fn=ask,
        inputs=[pending_query, chatbot, session_state, web_search_toggle, pending_request_id],
        outputs=chatbot,
        show_progress="minimal",
        trigger_mode="always_last",
        concurrency_limit=1,
        concurrency_id="chat_send",
        cancels=[send_response],
    )
    submit_response.then(
        fn=refresh_sessions_after_send,
        inputs=[session_state, recent_sessions_state],
        outputs=[recent_sessions_state, no_recent_chats] + session_btns
    )

    # New session action and refresh recent sessions list
    new_session_btn.click(
        fn=new_session,
        outputs=[session_state, chatbot, web_search_toggle],
    ).then(
        fn=lambda s: s,
        inputs=session_state,
        outputs=session_box,
    ).then(
        fn=refresh_sessions_list,
        inputs=[recent_sessions_state, session_state],
        outputs=[recent_sessions_state, no_recent_chats] + session_btns
    )

    # Wire up recent session buttons click handlers
    for i, btn in enumerate(session_btns):
        btn.click(
            make_session_click_handler(i),
            inputs=[recent_sessions_state, web_search_session_state],
            outputs=[session_state, session_box, chatbot, web_search_toggle],
        ).then(
            fn=refresh_sessions_list,
            inputs=[recent_sessions_state, session_state],
            outputs=[recent_sessions_state, no_recent_chats] + session_btns,
        )


# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Render (and most cloud hosts) assign the port via $PORT and require
    # binding to 0.0.0.0 so the service is reachable from outside.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=False,
        theme=THEME,
        css=CSS,
    )
