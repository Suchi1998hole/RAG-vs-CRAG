import os
from tqdm import tqdm
import pandas as pd

from src.rag import rag_answer
from src.crag import crag_answer
from src.metrics import measure, to_dataframe
from src.utils.logger import logger

QUERIES = [
    "Explain the role of alpha band in EEG-based motor imagery.",
    "Explain the role of alpha band in EEG-based motor imagery.",
    "What is the difference between delta, theta, alpha, beta, and gamma bands?",
    "How can S3 default encryption protect data at rest?",
    "How can S3 default encryption protect data at rest?",
    "Describe how VPC peering works in AWS.",
    "Describe how VPC peering works in AWS.",
    "Describe how VPC peering works in AWS.",
    "What are the limitations of RAG architectures?",
    "What are the limitations of RAG architectures?",
]

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def run_benchmark():
    rag_records = []
    crag_records = []

    logger.info("Running RAG baseline...")
    for q in tqdm(QUERIES, desc="RAG"):
        _, m = measure(rag_answer, q)
        m["query"] = q
        m["mode"] = "RAG"
        rag_records.append(m)

    logger.info("Running CAG (with cache)...")
    for q in tqdm(QUERIES, desc="CRAG"):
        _, m = measure(crag_answer, q)
        m["query"] = q
        m["mode"] = "CRAG"
        crag_records.append(m)

    df_rag = to_dataframe(rag_records)
    df_crag = to_dataframe(crag_records)


    df = pd.concat(
        [df_rag, df_crag.reset_index(drop=True)],
        ignore_index=True
    )


    out_path = os.path.join(OUT_DIR, "rag_vs_crag_metrics.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved benchmark results to {out_path}")

    # Simple console summary
    summary = df.groupby("mode")[["latency_ms", "tokens_total", "mem_delta_bytes"]].mean()
    logger.info(f"\n{summary}")

    return df


if __name__ == "__main__":
    run_benchmark()
