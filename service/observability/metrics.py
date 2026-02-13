# service/observability/metrics.py
from __future__ import annotations

from typing import Dict, Any

from fastapi import FastAPI
from prometheus_client import Counter, Histogram

# -----------------------------
# Low-level rules (important):
# - Keep labels LOW-cardinality (route, method, status are OK)
# - Never label by request_id, user_id, question text, IP, etc.
# - Use histograms for latency so p95 is easy to compute
# -----------------------------


def instrument_metrics(app: FastAPI) -> Dict[str, Any]:
    """
    Creates Prometheus metrics used by the API middleware and RAG pipeline.

    Demo note: this keeps all metric definitions together for quick wiring.

    Returns a dict of metric objects so callers can do:
      metrics["http_latency"].labels(path="/ask").observe(0.123)
      metrics["http_requests"].labels(method="POST", path="/ask", status="200").inc()
      metrics["ask_latency"].observe(0.456)
      metrics["ask_requests"].inc()
      metrics["rag_stage_latency"].labels(stage="retrieval").observe(0.12)
      metrics["rag_errors"].labels(error_type="TimeoutError").inc()
    """

    # Total HTTP requests by route/method/status
    http_requests = Counter(
        "http_requests_total",
        "Total HTTP requests",
        labelnames=("method", "path", "status"),
    )

    # End-to-end request latency by route
    http_latency = Histogram(
        "http_request_latency_seconds",
        "HTTP request latency in seconds",
        labelnames=("path",),
        # Tuned for RAG-ish latencies; adjust after measuring
        buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 21),
    )

    # Convenience: overall /ask latency and count (no labels => easy p95 on /ask)
    ask_requests = Counter(
        "ask_requests_total",
        "Total /ask requests",
    )

    ask_latency = Histogram(
        "ask_latency_seconds",
        "End-to-end /ask latency in seconds",
        buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 21),
    )

    # RAG stage timings (retrieval / rerank / generate / postprocess, etc.)
    rag_stage_latency = Histogram(
        "rag_stage_latency_seconds",
        "Latency by RAG stage",
        labelnames=("stage",),
        buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 3, 5, 8),
    )

    # RAG error counts (keep error_type bounded; don't put message text here)
    rag_errors = Counter(
        "rag_errors_total",
        "RAG errors by type",
        labelnames=("error_type",),
    )

    return {
        "http_requests": http_requests,
        "http_latency": http_latency,
        "ask_requests": ask_requests,
        "ask_latency": ask_latency,
        "rag_stage_latency": rag_stage_latency,
        "rag_errors": rag_errors,
    }
