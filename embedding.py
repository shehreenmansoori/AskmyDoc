from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
import os

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

embedding_model = MistralAIEmbeddings()

def store_documents(documents, collection_name="documents"):
    QdrantVectorStore.from_documents(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        embedding=embedding_model,
        documents=documents,
        collection_name=collection_name, 
    )
    return len(documents)


def store_web_documents(web_results, query):
    web_documents = []

    for result in web_results:
        title = result.get("title","")
        snippet = result.get("snippet","")
        link = result.get("link","")

        doc = Document(
            page_content=f"{title}\n{snippet}",
            metadata={
                "source_url": link,
                "query": query,
            },
        )
        web_documents.append(doc)

    return store_documents(
        web_documents,
        collection_name="web_documents"
    )