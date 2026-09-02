from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

embedding_model = MistralAIEmbeddings()


def get_retreiver(collection_name="documents"):
    try:
        vectorstore = QdrantVectorStore.from_existing_collection(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            embedding=embedding_model,
            collection_name=collection_name,
        )
    except Exception:
        # The collection doesn't exist yet (nothing has been uploaded),
        # so there is nothing to retrieve. Return None instead of crashing.
        return None

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,  
            "lambda_mult": 0.5,
        },
    )

def get_pdf_retreiver():
    return get_retreiver(collection_name="documents")

def get_web_retreiver():
    return get_retreiver(collection_name="web_documents")