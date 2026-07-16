# RAG Retrieval Script
# Given a natural language query, embeds it and retrieves the top-k
# most relevant chunks from the Chroma vector DB built by ingest.py

import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "usda_programs"


def get_collection():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    return collection


def retrieve(query: str, top_k: int = 3):
    # return the top_k most relevant chunks for a given query
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    # results is a dict with lists nested one level (one per query)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved.append({
            "text": doc,
            "source": meta.get("source"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist  # lower = more similar
        })

    return retrieved


def format_context(retrieved_chunks):
    # format retrieved chunks into a context block for the LLM prompt
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n\n"
    return context


if __name__ == "__main__":
    # quick manual test — run this file directly to try a query
    test_query = "Am I eligible for wildfire disaster assistance?"

    print(f"Query: {test_query}\n")
    results = retrieve(test_query, top_k=3)

    for i, r in enumerate(results):
        print(f"--- Result {i+1} (distance: {r['distance']:.4f}) ---")
        print(f"Source: {r['source']}")
        print(f"Text: {r['text'][:300]}...")
        print()