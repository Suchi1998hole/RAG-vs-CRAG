import os
from tqdm import tqdm
from src.config import DOCUMENTS_DIR, CLASS_NAME
from src.weaviate_client import ensure_schema
from src.openai_client import embed_text
from src.utils.logger import logger


CHUNK_SIZE = 700   
CHUNK_OVERLAP = 150


def iter_files():
    """Yield all .txt and .md files inside DOCUMENTS_DIR recursively."""
    for root, _, files in os.walk(DOCUMENTS_DIR):
        for f in files:
            if f.lower().endswith((".txt", ".md")):
                yield os.path.join(root, f)


def chunk_text(text: str):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def load_documents():
    """Load documents, embed them, and insert into Weaviate."""
    client = ensure_schema()
    collection = client.collections.get(CLASS_NAME)

    logger.info(f"Loading documents from {DOCUMENTS_DIR}")

    all_files = list(iter_files())
    if not all_files:
        logger.warning("No text or markdown files found in DOCUMENTS_DIR.")
        return

    for path in tqdm(all_files, desc="Indexing docs"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text)

        for c in chunks:
            vec = embed_text(c)
            collection.data.insert(
                properties={"text": c},
                vector=vec
            )

    logger.info("Document indexing complete.")
    client.close()
    print("Connection closed cleanly.")


if __name__ == "__main__":
    load_documents()
