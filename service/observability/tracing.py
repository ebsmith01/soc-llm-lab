"""
OpenTelemetry tracing setup for the SOC LLM API.
captures timing + metadata for incoming HTTP requests and RAG stages.

- Uses OTLP HTTP exporter if OTEL_EXPORTER_OTLP_ENDPOINT is set
- Instruments FastAPI automatically
- Safe to call once at startup (idempotent)
"""

from __future__ import annotations

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased


def setup_tracing(
    app=None,
    service_name: str = "soc-llm-api",
) -> Optional[trace.Tracer]:
    """
    Initialize OpenTelemetry tracing.

    Environment variables:
      - OTEL_EXPORTER_OTLP_ENDPOINT (e.g. http://localhost:4318)
      - OTEL_TRACES_SAMPLE_RATIO (default: 0.1)

    Returns:
      A tracer instance if tracing is enabled, else None.
    """

    # Demo note: exits early with a no-op tracer when no exporter is configured.
    # Prevent double-initialization (important for reload/tests)
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return trace.get_tracer(__name__)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    sample_ratio = float(os.getenv("OTEL_TRACES_SAMPLE_RATIO", "0.1"))

    # If no exporter endpoint, install a no-op provider
    if not endpoint:
        trace.set_tracer_provider(TracerProvider())
        return None

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",
        }
    )

    sampler = ParentBased(TraceIdRatioBased(sample_ratio))

    provider = TracerProvider(
        resource=resource,
        sampler=sampler,
    )

    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    # Auto-instrument FastAPI if app provided
    if app is not None:
        FastAPIInstrumentor.instrument_app(app)

    return trace.get_tracer(__name__)
