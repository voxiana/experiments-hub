"""
Optional telemetry wrappers for OpenTelemetry and Prometheus.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

OTEL_ENABLED = False
PROM_ENABLED = False
prom_eval_latency = None
prom_eval_success_count = None
prom_eval_failure_count = None


def _truthy(value: str | None) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def init_telemetry() -> None:
    global OTEL_ENABLED
    global PROM_ENABLED
    global prom_eval_latency
    global prom_eval_success_count
    global prom_eval_failure_count

    if _truthy(os.getenv("ENABLE_OTEL")):
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            resource = Resource.create({"service.name": "voice-model-eval-runner"})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            trace.set_tracer_provider(provider)
            OTEL_ENABLED = True
        except Exception:
            OTEL_ENABLED = False

    if _truthy(os.getenv("ENABLE_PROMETHEUS")):
        try:
            from prometheus_client import Counter, Histogram, start_http_server

            prom_eval_latency = Histogram(
                "voice_eval_latency_ms",
                "Latency per backend/case",
                labelnames=("backend_id", "case_id"),
            )
            prom_eval_success_count = Counter(
                "voice_eval_success_total",
                "Successful syntheses",
                labelnames=("backend_id",),
            )
            prom_eval_failure_count = Counter(
                "voice_eval_failure_total",
                "Failed syntheses",
                labelnames=("backend_id",),
            )
            start_http_server(int(os.getenv("PROMETHEUS_PORT", "9108")))
            PROM_ENABLED = True
        except Exception:
            PROM_ENABLED = False


@contextmanager
def eval_span(name: str, attributes: dict[str, str]) -> Iterator[None]:
    if not OTEL_ENABLED:
        yield
        return

    from opentelemetry import trace

    tracer = trace.get_tracer("voice-model-eval")
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield


def record_metrics(backend_id: str, case_id: str, success: bool, latency_ms: float | None) -> None:
    if not PROM_ENABLED:
        return

    if latency_ms is not None and prom_eval_latency is not None:
        prom_eval_latency.labels(backend_id=backend_id, case_id=case_id).observe(latency_ms)

    if success and prom_eval_success_count is not None:
        prom_eval_success_count.labels(backend_id=backend_id).inc()
    if not success and prom_eval_failure_count is not None:
        prom_eval_failure_count.labels(backend_id=backend_id).inc()

