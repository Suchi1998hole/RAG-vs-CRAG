import json
import hashlib
import numpy as np
import redis
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    SEMANTIC_CACHE_THRESHOLD
)
from .openai_client import embed_text
from .utils.logger import logger


def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_exact(query: str):
    r = get_redis()
    key = f"exact:{_hash(query)}"
    val = r.get(key)
    if val:
        return json.loads(val)
    return None


def set_exact(query: str, data: dict, ttl: int = 3600):
    r = get_redis()
    key = f"exact:{_hash(query)}"
    r.setex(key, ttl, json.dumps(data))


def get_semantic(query: str):
    """
    Semantic cache:
    - We store a list "semantic_cache" with entries {embedding, answer, meta}.
    - At lookup, embed query and scan for cosine sim above threshold.
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
        logger.debug(f"Semantic cache hit (sim={best_sim:.3f})")
        return best_entry
    return None


def add_semantic(query: str, answer: str, usage, meta=None, max_entries: int = 500):
    r = get_redis()
    raw = r.get("semantic_cache")
    entries = json.loads(raw) if raw else []

    q_emb = embed_text(query)

    entry = {
        "embedding": q_emb,
        "answer": answer,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
        "meta": meta or {}
    }

    entries.append(entry)
    if len(entries) > max_entries:
        entries = entries[-max_entries:]

    r.set("semantic_cache", json.dumps(entries))
