from fastapi import FastAPI, UploadFile, File
from chunking import process_pdf
from embedding import store_documents,store_web_documents
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import os
from mongo_db import metadata_collection
from dotenv import load_dotenv
from datetime import datetime,timezone
import logging
import uuid
from langchain_community.tools import DuckDuckGoSearchResults
from retreiver import get_pdf_retreiver, get_web_retreiver
import re
from typing import TypedDict,Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)

logger = logging.getLogger(__name__)

load_dotenv()
 
app = FastAPI()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

@app.post("/documents/upload")
async def upload_file(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    documents = process_pdf(pdf_bytes)
    logger.info(f"CHUNKS CREATED:{len(documents)}")

    embeddings = store_documents(documents)
    logger.info(f"EMBDEDDINGS CREATED:{embeddings}")

    return {"chunks": len(documents),
            "embedding": embeddings
            }

llm = ChatMistralAI(model="mistral-small-latest")

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

# LLM will answer using live web search results.
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

@app.get("/create_session")
async def create_session():
    return{
        "session_id":str(uuid.uuid4())
    } 

@app.post("/ask")
async def ask_question(query:str,session_id:str,enable_web_search:bool=False):
    retriever = get_pdf_retreiver()

    previous_chats = list(
        metadata_collection.find(
            {"session_id":session_id}
        ).sort("timestamp",-1).limit(3)
    )

    history = ""
    for chat in reversed(previous_chats):
        history += f"User: {chat['query']}\n"
        history += f"Bot: {chat['answer']}\n"

    repeat_chat = metadata_collection.find_one(
        {
        "query":query
        },
        sort=[("timestamp",-1)]
    )
    logger.info(f"Repeat question checking: {'FOUND' if repeat_chat else 'NOT FOUND'}")

    #if the question was repeated previously
    if repeat_chat:
        judge_response = llm.invoke(judge_prompt.invoke({
            "question":query,
            "previous_answer":repeat_chat["answer"]
        }))

        #If answer is sufficient
        if "YES" in judge_response.content.upper():
            logger.info("Cached answer is sufficient to reuse.")
            return{
                "session_id": session_id,
                "answer": repeat_chat["answer"],
                "sources":repeat_chat.get("sources",[])
            }
        else: #if answer is not sufficient
            logger.info("Cached answer was not sufficient for the current query, regenerating answer.")

    # pdf
    docs = retriever.invoke(query)
    logger.info(f"DOCUMENTS RETRIEVED:{len(docs)}")
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = pdf_prompt.invoke({
        "history": history,
        "context": context,
        "question": query
    })
    response = llm.invoke(final_prompt)

    used_web_search = False
    sources=[]
    #If answer is present in pdf
    if "NOT_FOUND" not in response.content:
        sources = [
            {
                "source_number": i + 1, 
                "content": doc.page_content
            }
            for i, doc in enumerate(docs)
        ]

    #if answer is not present and web search is enabled
    elif enable_web_search:
        used_web_search = True
        logger.info("Answer not in PDF, searching the web.")

        search_tool =DuckDuckGoSearchResults(output_format="list")
        web_results = search_tool.invoke(query) #raw DuckDuckGo results

        store_web_documents(web_results, query) #save them into Qdrant
        logger.info("Web search stored in database") 
        
        web_retreiver = get_web_retreiver() #search web Qdrant collection
        web_docs = web_retreiver.invoke(query) #best matching web documents 

        web_context = "\n\n".join(doc.page_content for doc in web_docs) #text given to LLM

        final_prompt = web_prompt.invoke({
            "history": history,
            "web_results": web_context,
            "question": query
        })
        response = llm.invoke(final_prompt)
        logger.info("Web search answer generated for query.")

        sources = []
        for i, doc in enumerate(web_docs,start=1):
            sources.append({
                "source_number": i,
                "content": doc.page_content,
                "metadata": doc.metadata
            })
    #If answer was not found and web search is not enabled
    else:
        response.content = (
            "Couldn't find this in the uploaded document.Turn on enable_web_search to search the web."
        )

    metadata_collection.insert_one({
        "query": query,
        "session_id": session_id,
        "answer": response.content,
        "used_web_search": used_web_search,
        "timestamp": datetime.now(timezone.utc),
        "documents_retrieved": len(docs),
        "sources": sources,
    })

    return{
        "session_id":session_id,
        "answer":response.content,
        "used_web_search": used_web_search,
        "sources":sources,
    }     