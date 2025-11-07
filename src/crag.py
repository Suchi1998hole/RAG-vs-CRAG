from .cache_layer import (
    get_exact, set_exact,
    get_semantic, add_semantic
)
from .rag import rag_answer

def crag_answer(query: str):
    # 1. Exact cache
    exact = get_exact(query)
    if exact:
        exact["source"] = "exact_cache"
        return exact

    # 2. Semantic cache
    semantic = get_semantic(query)
    if semantic:
        return {
            "answer": semantic["answer"],
            "usage": semantic["usage"],
            "retrieved_docs": semantic.get("meta", {}).get("retrieved_docs", []),
            "source": "semantic_cache"
        }

    # 3. Fallback to RAG
    rag_res = rag_answer(query)
    rag_res["source"] = "rag_fresh"

    # Cache result
    set_exact(query, rag_res)
    add_semantic(
        query,
        rag_res["answer"],
        rag_res["usage"],
        meta={"retrieved_docs": rag_res["retrieved_docs"]}
    )

    return rag_res
