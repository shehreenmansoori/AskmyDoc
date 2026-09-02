import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

def process_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text() + "\n"
    chunks = splitter.split_text(text)

    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents
