from src.cache_layer import (
    get_exact, set_exact,
    get_semantic, add_semantic
)
from src.rag import rag_answer


def crag_answer(query: str):
    """
    Cached Retrieval-Augmented Generation (CRAG):
    1. Check exact cache
    2. Check semantic cache
    3. Run fresh RAG and update both caches
    """

    exact = get_exact(query)
    if exact:
        if isinstance(exact, str):
            import json
            try:
                exact = json.loads(exact)
            except Exception:
                exact = {"answer": exact, "usage": {}, "retrieved_docs": []}
        exact["source"] = "exact_cache"
        return exact

    semantic = get_semantic(query)
    if semantic:
        return {
            "answer": semantic.get("answer", ""),
            "usage": semantic.get("usage", {}),
            "retrieved_docs": semantic.get("meta", {}).get("retrieved_docs", []),
            "source": "semantic_cache"
        }

    rag_res = rag_answer(query)

    answer = rag_res.get("answer", "")
    usage = rag_res.get("usage", {})
    retrieved_docs = rag_res.get("retrieved_docs", [])

    safe_cache_data = {
        "answer": answer,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
            "total_tokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
        },
        "retrieved_docs": retrieved_docs,
        "source": "rag_fresh"
    }

    set_exact(query, safe_cache_data)
    add_semantic(
        query=query,
        answer=answer,
        usage=safe_cache_data["usage"],
        meta={"retrieved_docs": retrieved_docs}
    )

    return safe_cache_data
