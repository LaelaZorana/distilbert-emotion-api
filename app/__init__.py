"""Production inference service for the DistilBERT emotion classifier.

Wraps the fine-tuned ``LaelaZ/distilbert-emotion`` model (six emotions) behind a
FastAPI service with batching, Prometheus metrics, and health probes. The whole
service runs offline against a deterministic stub classifier when ``OFFLINE=1``,
so the API, demo, tests, and load test work with no model download.
"""
from __future__ import annotations

__version__ = "1.0.0"

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}

__all__ = ["__version__", "LABELS", "ID2LABEL", "LABEL2ID"]
