"""FastAPI application: routes, lifespan, and wiring.

Endpoints:
    GET  /            -> redirect to the demo UI
    GET  /healthz     -> readiness/liveness (200 once the model is loaded)
    GET  /metrics     -> Prometheus exposition
    POST /predict     -> single or batch classification (pydantic-validated)
    GET  /demo        -> static HTML demo that calls /predict

The model is loaded once in the lifespan startup and shared via ``app.state``;
the micro-batcher runs as a background task for the lifetime of the process.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app import __version__
from app.batching import MicroBatcher
from app.classifier import load_classifier
from app.config import get_settings
from app.metrics import metrics_middleware, render_latest
from app.schemas import (BatchPredictRequest, BatchPredictResponse,
                         HealthResponse, PredictRequest, PredictResponse,
                         Prediction)

logger = logging.getLogger(__name__)

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
STATIC_DIR = DEMO_DIR / "vendor"


def _build_prediction(text: str, dist: Dict[str, float]) -> Prediction:
    top = max(dist, key=dist.get)
    return Prediction(label=top, score=dist[top], probabilities=dist)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "starting emotion-api v%s (offline=%s, model_id=%s)",
        __version__, settings.offline, settings.model_id,
    )
    classifier = load_classifier(settings)
    batcher = MicroBatcher(
        classifier,
        max_batch_size=settings.max_batch_size,
        max_delay_ms=settings.batch_max_delay_ms,
    )
    await batcher.start()

    app.state.settings = settings
    app.state.classifier = classifier
    app.state.batcher = batcher
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        await batcher.stop()
        logger.info("emotion-api stopped")


def create_app() -> FastAPI:
    """Application factory (one per process; tests build their own)."""
    app = FastAPI(
        title="DistilBERT Emotion API",
        version=__version__,
        summary="Production inference service for the LaelaZ/distilbert-emotion model.",
        description=(
            "Classify a sentence into one of six emotions "
            "(sadness, joy, love, anger, fear, surprise) with full per-class "
            "probabilities. Batched, observable, and runnable fully offline."
        ),
        lifespan=lifespan,
    )
    app.middleware("http")(metrics_middleware)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/demo")

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz(request: Request) -> HealthResponse:
        """Readiness + liveness. 503 until the model is loaded and the batcher runs."""
        state = request.app.state
        if not getattr(state, "ready", False):
            raise HTTPException(status_code=503, detail="not ready")
        settings = state.settings
        return HealthResponse(
            status="ok",
            backend=state.classifier.backend,
            model_id=settings.model_id,
            offline=settings.offline,
            version=__version__,
        )

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics():
        return render_latest()

    @app.post(
        "/predict",
        tags=["inference"],
        summary="Classify one sentence or a batch.",
        responses={
            200: {
                "description": "Single prediction (PredictResponse) or batch "
                "(BatchPredictResponse), matching the request shape."
            }
        },
    )
    async def predict(request: Request, body: Dict[str, Any]) -> Any:
        """Accept ``{text: str}`` or ``{texts: [str, ...]}`` and classify.

        The body is parsed manually so one endpoint can serve both shapes and
        return the matching response shape. Validation errors surface as 422.
        """
        batcher: MicroBatcher = request.app.state.batcher

        is_single = "text" in body
        is_batch = "texts" in body
        if is_single == is_batch:  # neither, or both
            raise HTTPException(
                status_code=422,
                detail="provide exactly one of 'text' (string) or 'texts' (list of strings)",
            )

        try:
            if is_single:
                req = PredictRequest(**body)
                dist = await batcher.submit(req.text)
                pred = _build_prediction(req.text, dist)
                return PredictResponse(**pred.model_dump())
            req_b = BatchPredictRequest(**body)
            dists = await batcher.submit_many(req_b.texts)
            preds: List[Prediction] = [
                _build_prediction(t, d) for t, d in zip(req_b.texts, dists)
            ]
            return BatchPredictResponse(predictions=preds)
        except ValidationError as exc:
            # Translate pydantic errors into FastAPI's standard 422 envelope.
            # Strip the non-JSON-serializable bits pydantic attaches (the raw
            # exception object under `ctx`, and the docs URL).
            detail = [
                {k: v for k, v in err.items() if k not in ("ctx", "url")}
                for err in exc.errors()
            ]
            raise HTTPException(status_code=422, detail=detail)

    @app.get("/demo", tags=["ui"], include_in_schema=False)
    async def demo() -> FileResponse:
        return FileResponse(DEMO_DIR / "index.html")

    # Vendored, offline front-end assets (Tailwind Play CDN bundle). Mounting a
    # directory keeps the demo page fully network-free: no external CDN at runtime.
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
