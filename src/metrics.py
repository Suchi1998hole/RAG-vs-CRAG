import time
import psutil
import pandas as pd

process = psutil.Process()


def measure(fn, *args, **kwargs):
    """
    Measures:
    - wall time
    - cpu_percent delta (approx)
    - memory (rss) before & after
    - plus any 'usage' dict from fn result
    """
    mem_before = process.memory_info().rss
    cpu_before = psutil.cpu_percent(interval=None)

    t0 = time.time()
    result = fn(*args, **kwargs)
    t1 = time.time()

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss

    latency = (t1 - t0) * 1000.0  # ms
    cpu = max(0.0, cpu_after - cpu_before)
    mem_diff = mem_after - mem_before

    usage = getattr(result.get("usage", None), "__dict__", result.get("usage", {}))

    metrics = {
        "latency_ms": latency,
        "cpu_delta": cpu,
        "mem_delta_bytes": mem_diff,
        "tokens_prompt": getattr(result.get("usage", None), "prompt_tokens", usage.get("prompt_tokens", None)),
        "tokens_completion": getattr(result.get("usage", None), "completion_tokens", usage.get("completion_tokens", None)),
        "tokens_total": getattr(result.get("usage", None), "total_tokens", usage.get("total_tokens", None)),
        "source": result.get("source", "unknown")
    }

    return result, metrics


def to_dataframe(records):
    return pd.DataFrame(records)
