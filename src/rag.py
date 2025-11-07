from .openai_client import embed_text, chat_completion
from .weaviate_client import get_weaviate_client
from .config import CLASS_NAME, TOP_K

def rag_answer(query: str):
    client = get_weaviate_client()
    q_vec = embed_text(query)

    result = (
        client.query
        .get(CLASS_NAME, ["text"])
        .with_near_vector({"vector": q_vec})
        .with_limit(TOP_K)
        .do()
    )

    docs = result["data"]["Get"].get(CLASS_NAME, [])
    context = "\n\n".join(d["text"] for d in docs)

    prompt = (
        "You are a helpful assistant. Use ONLY the context to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    answer, usage = chat_completion(prompt)
    return {
        "answer": answer,
        "usage": usage,
        "retrieved_docs": docs
    }
