from src.openai_client import embed_text, chat_completion
from src.weaviate_client import get_weaviate_client
from src.config import CLASS_NAME, TOP_K


def rag_answer(query: str):
    """
    Performs Retrieval-Augmented Generation (RAG) using Weaviate v4.17+ API.
    Steps:
      1. Embed the query using OpenAI
      2. Retrieve top-k semantically similar docs from Weaviate
      3. Pass them to GPT for contextual answering
    """
    client = get_weaviate_client()
    collection = client.collections.get(CLASS_NAME)
    q_vec = embed_text(query)

    result = collection.query.near_vector(
        near_vector=q_vec,
        limit=TOP_K,
        return_properties=["text"]
    )

    docs = [o.properties["text"] for o in result.objects]
    context = "\n\n".join(docs)

    prompt = (
        "You are a helpful assistant. Use ONLY the context to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    answer, usage = chat_completion(prompt)
    client.close()  

    return {
        "answer": answer,
        "usage": usage,
        "retrieved_docs": docs
    }
