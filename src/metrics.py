import time
import psutil
import pandas as pd
import json

process = psutil.Process()


def safe_get_usage(result):
    """
    Safely extract and normalize 'usage' from function result.
    Converts strings or objects to plain dicts.
    """
    usage = result.get("usage", {})
    if usage is None:
        return {}

    if isinstance(usage, str):
        try:
            usage = json.loads(usage)
        except Exception:
            usage = {}

    if hasattr(usage, "__dict__"):
        usage = usage.__dict__

    if not isinstance(usage, dict):
        usage = {}

    return usage


def measure(fn, *args, **kwargs):
    """
    Measures:
    - Wall-clock time (latency)
    - CPU usage delta (%)
    - Memory delta (RSS)
    - OpenAI token usage (prompt, completion, total)
    - Source label (fresh or semantic cache)
    """
    mem_before = process.memory_info().rss
    cpu_before = psutil.cpu_percent(interval=None)

    t0 = time.time()
    result = fn(*args, **kwargs)
    t1 = time.time()

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss

    latency = (t1 - t0) * 1000.0  # milliseconds
    cpu_delta = max(0.0, cpu_after - cpu_before)
    mem_delta = mem_after - mem_before

    usage = safe_get_usage(result)

    metrics = {
        "latency_ms": round(latency, 2),
        "cpu_delta": cpu_delta,
        "mem_delta_bytes": mem_delta,
        "tokens_prompt": usage.get("prompt_tokens", 0),
        "tokens_completion": usage.get("completion_tokens", 0),
        "tokens_total": usage.get("total_tokens", 0),
        "source": result.get("source", "unknown"),
    }

    return result, metrics


def to_dataframe(records):
    """
    Convert list of metric dicts to a pandas DataFrame.
    """
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)
