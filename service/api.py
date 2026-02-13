from __future__ import annotations

# /health is for uptime checks, /metrics feeds Prometheus for monitoring, and /ask is the main RAG query endpoint.
# start-up telemetry (logging + tracing) is wired in create_app() and the middleware.

import time
import uuid
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from service import schemas
from service.rag_service import RagService
from service.observability.logging import setup_logging
from service.observability.metrics import instrument_metrics
from service.observability.tracing import setup_tracing

log = structlog.get_logger()


def create_app() -> FastAPI:
    setup_logging()
    setup_tracing()

    # Demo note: minimal FastAPI setup that wires telemetry and a single RAG endpoint.
    app = FastAPI(title="SOC LLM Lab API", version="0.1.0")
    metrics = instrument_metrics(app)
    svc = RagService()

    @app.middleware("http")
    async def request_telemetry(request: Request, call_next):
        # Correlation id (lets you tie logs/traces/metrics together)
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            # Structured error log (stack trace)
            log.exception(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error_type=type(e).__name__,
            )
            raise
        finally:
            duration = time.perf_counter() - start
            latency_ms = round(duration * 1000, 2)

            # Metrics: keep labels LOW-cardinality (route + status + method)
            if metrics:
                # If your instrument_metrics already defines these keys, use them.
                # (If not, adjust instrument_metrics to match.)
                metrics["http_latency"].labels(path=request.url.path).observe(duration)
                metrics["http_requests"].labels(
                    method=request.method,
                    path=request.url.path,
                    status=str(status_code),
                ).inc()

                # Optional convenience counters for /ask
                if request.url.path == "/ask":
                    metrics["ask_latency"].observe(duration)
                    metrics["ask_requests"].inc()

            # One INFO log per request
            log.info(
                "request_summary",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=status_code,
                latency_ms=latency_ms,
            )

    @app.get("/health", summary="Health check")
    async def health() -> JSONResponse:
        # Liveness (fast)
        return JSONResponse({"status": "ok"})

    @app.get("/metrics", summary="Prometheus metrics")
    async def metrics_endpoint() -> Response:
        # Prometheus scrape endpoint
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    @app.post("/ask", response_model=schemas.QueryResponse, summary="Run RAG query")
    async def ask(payload: schemas.QueryRequest, request: Request) -> schemas.QueryResponse:
        # NOTE: svc.answer is likely blocking (retrieval, embeddings, LLM calls).
        # Use threadpool so the event loop stays responsive.
        result = await run_in_threadpool(
            svc.answer,
            payload.question,
            use_local_lora=payload.use_local_lora,
            top_k=payload.top_k,
            alpha=payload.alpha,
        )

        return schemas.QueryResponse(**result)

    return app


app = create_app()
