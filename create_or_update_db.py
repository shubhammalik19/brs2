import os
import shutil
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration from .env
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
DB_NAME = os.getenv("DB_NAME", "BRS_DB")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

# Delete existing ChromaDB if it exists
if os.path.exists(DB_NAME):
    shutil.rmtree(DB_NAME)
    print(f"Deleted existing database: {DB_NAME}")

# Initialize Ollama Embeddings
embedding_model = OllamaEmbeddings(model=EMBED_MODEL)

# Initialize ChromaDB
vector_store = Chroma(persist_directory=DB_NAME, embedding_function=embedding_model)

# Function to extract text from PDFs
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)

# Process PDFs and store in ChromaDB
def save_pdfs():
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found.")
        return

    docs = []
    for pdf_file in pdf_files:
        text = extract_text(os.path.join(DATA_FOLDER, pdf_file))
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
        docs.extend({"text": chunk, "metadata": {"source": pdf_file}} for chunk in chunks)

    if docs:
        vector_store.add_texts([doc["text"] for doc in docs], metadatas=[doc["metadata"] for doc in docs])
        print(f"Saved {len(docs)} chunks to ChromaDB ({DB_NAME}).")

if __name__ == "__main__":
    save_pdfs()
    print("PDF data successfully saved to ChromaDB!")
