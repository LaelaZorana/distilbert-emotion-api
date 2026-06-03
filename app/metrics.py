"""Prometheus instrumentation.

Exposes the four signals a serving SLO is built on: request count, in-flight
gauge, latency histogram, and error count. A middleware records HTTP-level
metrics for every route; the inference path additionally records model latency
and batch sizes so you can separate "the model is slow" from "the network /
framework is slow".

All metrics live in a module-level registry that ``/metrics`` renders, so the
collectors are created exactly once even if the app is imported repeatedly
(as happens under pytest).
"""
from __future__ import annotations

import time
from typing import Callable

from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter,
                               Gauge, Histogram, generate_latest)
from starlette.requests import Request
from starlette.responses import Response

REGISTRY = CollectorRegistry()

# Latency buckets tuned for a CPU transformer: sub-ms (stub) up to a couple of
# seconds (cold real model / large batch).
_LATENCY_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0,
)

REQUEST_COUNT = Counter(
    "emotion_api_requests_total",
    "Total HTTP requests processed, by method, path template, and status.",
    ["method", "path", "status"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "emotion_api_request_latency_seconds",
    "End-to-end HTTP request latency in seconds, by method and path template.",
    ["method", "path"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

ERROR_COUNT = Counter(
    "emotion_api_errors_total",
    "Total responses with a 5xx status, by method and path template.",
    ["method", "path"],
    registry=REGISTRY,
)

IN_PROGRESS = Gauge(
    "emotion_api_requests_in_progress",
    "Number of HTTP requests currently being served.",
    registry=REGISTRY,
)

INFERENCE_LATENCY = Histogram(
    "emotion_api_inference_latency_seconds",
    "Model forward-pass latency in seconds (excludes HTTP / validation).",
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

BATCH_SIZE = Histogram(
    "emotion_api_inference_batch_size",
    "Number of texts coalesced into a single model forward pass.",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
    registry=REGISTRY,
)


def _route_template(request: Request) -> str:
    """Use the route pattern (not the raw URL) so paths don't explode cardinality."""
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """Record count, latency, in-flight, and errors for every request."""
    method = request.method
    start = time.perf_counter()
    IN_PROGRESS.inc()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = _route_template(request)
        IN_PROGRESS.dec()
        REQUEST_LATENCY.labels(method, path).observe(elapsed)
        REQUEST_COUNT.labels(method, path, str(status_code)).inc()
        if status_code >= 500:
            ERROR_COUNT.labels(method, path).inc()


def render_latest() -> Response:
    """Render the registry in Prometheus text exposition format."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
