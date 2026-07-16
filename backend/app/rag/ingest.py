# RAG Ingest Script
# Loads PDFs from data/usda_docs/, chunks them, embeds them with a free
# local model, and stores them in a persistent Chroma vector DB

import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCS_DIR = "../../../data/usda_docs" # adjust relative path if needed
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db") # local persistent vector DB
COLLECTION_NAME = "usda_programs"

CHUNK_SIZE = 800 # characters per chunk
CHUNK_OVERLAP = 150 # overlap so context isn't cut mid-sentence


def load_pdf_text(filepath: str) -> str:
    # extract raw text from a single PDF
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, source: str):
    # split raw text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


def main():
    if not os.path.isdir(DOCS_DIR):
        print(f"ERROR: docs folder not found at {DOCS_DIR}")
        print("Create it and add your USDA PDFs first.")
        return

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {DOCS_DIR}. Add some and re-run.")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting ingest...")

    # local embedding model
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # fresh collection each run (delete if it exists, then recreate)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    all_ids = []
    all_documents = []
    all_metadatas = []

    for pdf_file in pdf_files:
        filepath = os.path.join(DOCS_DIR, pdf_file)
        print(f"  Processing: {pdf_file}")

        text = load_pdf_text(filepath)
        if not text.strip():
            print(f"    WARNING: no extractable text in {pdf_file}, skipping")
            continue

        chunks = chunk_text(text, pdf_file)
        print(f"    -> {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{pdf_file}_chunk{i}"
            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_metadatas.append({"source": pdf_file, "chunk_index": i})

    if not all_documents:
        print("No chunks were created. Check your PDFs have extractable text.")
        return

    # add everything to Chroma in one batch
    collection.add(
        ids=all_ids,
        documents=all_documents,
        metadatas=all_metadatas
    )

    print(f"\nDone. Ingested {len(all_documents)} chunks from {len(pdf_files)} documents.")
    print(f"Vector DB persisted at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()