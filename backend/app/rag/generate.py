# RAG Generation Script
# Retrieves relevant chunks for a query, then uses a local Ollama model
# to generate a grounded answer based only on that retrieved context

import ollama
from retrieve import retrieve, format_context

MODEL_NAME = "llama3.2"

SYSTEM_PROMPT = """You are Earthworm AI, a helpful assistant for farmers navigating USDA programs.

Answer the user's question using ONLY the information provided in the context below.
If the context does not contain enough information to answer, say so clearly instead of guessing.
Always mention which source document(s) your answer is based on.
Keep answers concise and practical for a farmer to act on."""


def generate_answer(query: str, top_k: int = 3):
    # step 1: retrieve relevant chunks
    retrieved_chunks = retrieve(query, top_k=top_k)
    context = format_context(retrieved_chunks)

    # step 2: build the prompt
    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    # step 3: call the local LLM
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "sources": list(set(c["source"] for c in retrieved_chunks)),
        "retrieved_chunks": retrieved_chunks
    }


if __name__ == "__main__":
    test_query = "How do I report my crop acreage?"
    print(f"Query: {test_query}\n")
    result = generate_answer(test_query)

    print("=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"- {s}")