import os
from textwrap import wrap
from tqdm import tqdm

from .config import DOCUMENTS_DIR, CLASS_NAME
from .weaviate_client import ensure_schema
from .openai_client import embed_text
from .utils.logger import logger


CHUNK_SIZE = 700   # chars
CHUNK_OVERLAP = 150


def iter_files():
    for root, _, files in os.walk(DOCUMENTS_DIR):
        for f in files:
            if f.lower().endswith((".txt", ".md")):
                yield os.path.join(root, f)


def chunk_text(text: str):
    # simple overlapping chunks
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def load_documents():
    client = ensure_schema()
    logger.info(f"Loading documents from {DOCUMENTS_DIR}")

    for path in tqdm(list(iter_files()), desc="Indexing docs"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = chunk_text(text)
        for c in chunks:
            vec = embed_text(c)
            client.data_object.create(
                data={"text": c},
                class_name=CLASS_NAME,
                vector=vec
            )

    logger.info("Document indexing complete.")


if __name__ == "__main__":
    load_documents()
