import os
import json
import hashlib
import numpy as np
import redis
from sklearn.metrics.pairwise import cosine_similarity
import json


from src.config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    SEMANTIC_CACHE_THRESHOLD
)
from .openai_client import embed_text
from .utils.logger import logger


def get_redis():
    """
    REDIS connection
    """
    redis_password = os.getenv("REDIS_PASSWORD", None)

    return redis.Redis(
        host=REDIS_HOST,
        port=int(REDIS_PORT),
        db=int(REDIS_DB),
        password=redis_password,    
        decode_responses=True
    )


def _hash(text: str) -> str:
    """
    SHA-256 hash of a query for consistent cache keys.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_exact(query: str):
    """
    Retrieve exact query-response from cache if it exists.
    """
    r = get_redis()
    key = f"exact:{_hash(query)}"
    val = r.get(key)
    if val:
        logger.debug(f"Exact cache hit for query: {query[:50]}...")
        return json.loads(val)
    return None



def set_exact(query: str, data: dict, ttl: int = 3600):
    """Cache exact query-answer pairs, safely converting non-JSON types."""
    r = get_redis()
    key = f"exact:{_hash(query)}"

    try:
        val = json.dumps(data)
    except TypeError:
        def safe_convert(obj):
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)

        val = json.dumps(data, default=safe_convert)

    r.setex(key, ttl, val)



def get_semantic(query: str):
    """
    Semantic cache:
    - Stores embeddings and responses
    - Uses cosine similarity 
    """
    r = get_redis()
    raw = r.get("semantic_cache")
    if not raw:
        return None

    entries = json.loads(raw)
    q_vec = np.array(embed_text(query)).reshape(1, -1)

    best_sim = 0.0
    best_entry = None

    for e in entries:
        emb = np.array(e["embedding"]).reshape(1, -1)
        sim = float(cosine_similarity(q_vec, emb)[0][0])
        if sim > best_sim:
            best_sim = sim
            best_entry = e

    if best_entry and best_sim >= SEMANTIC_CACHE_THRESHOLD:
        logger.debug(f"Semantic cache hit (sim={best_sim:.3f}) for query: {query[:50]}...")
        return best_entry
    return None


def add_semantic(query: str, answer: str, usage, meta=None, max_entries: int = 500):
    """
    Adds a new semantic cache entry (embedding + answer + metadata).
    Keeps the last 'max_entries' items.
    """
    r = get_redis()
    raw = r.get("semantic_cache")
    entries = json.loads(raw) if raw else []

    q_emb = embed_text(query)

    entry = {
        "embedding": q_emb,
        "answer": answer,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        "meta": meta or {}
    }

    entries.append(entry)
    if len(entries) > max_entries:
        entries = entries[-max_entries:]  

    r.set("semantic_cache", json.dumps(entries))
    logger.debug(f"Added semantic cache entry. Total entries: {len(entries)}")
