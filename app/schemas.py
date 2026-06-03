"""Request/response models with validation (pydantic v2).

The schemas double as the API contract and the validation layer: empty strings,
oversized inputs, and oversized batches are rejected with 422 before they ever
reach the model.
"""
from __future__ import annotations

from typing import Dict, List, Union

from pydantic import BaseModel, Field, field_validator

from app import LABELS
from app.config import get_settings

# Resolved once for schema bounds; the running app re-reads settings at request
# time for behavior, but the static field constraints are fine to fix here.
_MAX_TEXT = get_settings().max_text_length
_MAX_BATCH = get_settings().max_batch_size


class PredictRequest(BaseModel):
    """A single prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_TEXT,
        description="The sentence to classify.",
        examples=["i can't stop smiling, today went better than i ever hoped"],
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace-only")
        return v


class BatchPredictRequest(BaseModel):
    """A batch prediction request."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=_MAX_BATCH,
        description="One to MAX_BATCH_SIZE sentences to classify.",
        examples=[["i feel wonderful today", "i am so scared right now"]],
    )

    @field_validator("texts")
    @classmethod
    def _items_valid(cls, v: List[str]) -> List[str]:
        for i, t in enumerate(v):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"texts[{i}] must be a non-blank string")
            if len(t) > _MAX_TEXT:
                raise ValueError(f"texts[{i}] exceeds max_text_length ({_MAX_TEXT})")
        return v


class Prediction(BaseModel):
    """One classified sentence."""

    label: str = Field(..., description="Top emotion (argmax of the distribution).")
    score: float = Field(..., ge=0.0, le=1.0, description="Probability of the top label.")
    probabilities: Dict[str, float] = Field(
        ..., description="Full probability distribution over all labels (sums to ~1.0)."
    )


class PredictResponse(Prediction):
    """Response for a single ``/predict`` call (one prediction, flattened)."""


class BatchPredictResponse(BaseModel):
    """Response for a batch ``/predict`` call."""

    predictions: List[Prediction]


# Accept either shape on POST /predict so a single sentence and a batch share one
# endpoint without the caller juggling two URLs.
PredictBody = Union[PredictRequest, BatchPredictRequest]


class HealthResponse(BaseModel):
    """Readiness / liveness payload."""

    status: str = Field(..., examples=["ok"])
    backend: str = Field(..., description="Active classifier backend.", examples=["stub"])
    model_id: str
    offline: bool
    labels: List[str] = Field(default_factory=lambda: list(LABELS))
    version: str
