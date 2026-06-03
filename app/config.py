"""Runtime configuration, sourced from environment variables.

A tiny, dependency-free settings object (no pydantic-settings needed) so the
service stays lean and the import graph is trivial. Everything has a sane default
so the service boots with zero configuration in offline mode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable view of the environment, resolved once at import time.

    Attributes:
        offline: When True, the real model is replaced by a deterministic stub
            so nothing is downloaded. Defaults to True so a fresh checkout runs
            without network access; set ``OFFLINE=0`` for a real model load.
        model_id: Hugging Face repo (or local path) used when ``offline`` is False.
        max_batch_size: Hard cap on items accepted in one ``/predict`` call and
            the most the micro-batcher will coalesce into a single forward pass.
        max_text_length: Reject any single input longer than this many characters
            (cheap guard before tokenization).
        batch_max_delay_ms: How long the micro-batcher waits to fill a batch
            before flushing. Small values favor latency; larger favor throughput.
        log_level: Root log level for the service.
    """

    offline: bool = field(default_factory=lambda: _as_bool(os.getenv("OFFLINE"), True))
    model_id: str = field(default_factory=lambda: os.getenv("MODEL_ID", "LaelaZ/distilbert-emotion"))
    max_batch_size: int = field(default_factory=lambda: _as_int(os.getenv("MAX_BATCH_SIZE"), 64))
    max_text_length: int = field(default_factory=lambda: _as_int(os.getenv("MAX_TEXT_LENGTH"), 2000))
    batch_max_delay_ms: float = field(default_factory=lambda: _as_float(os.getenv("BATCH_MAX_DELAY_MS"), 5.0))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def labels(self) -> List[str]:
        from app import LABELS

        return list(LABELS)


def get_settings() -> Settings:
    """Return a freshly-resolved Settings.

    Re-read each call rather than caching a singleton, so tests can flip
    environment variables with ``monkeypatch`` and see the change.
    """
    return Settings()
